
import json
import cv2
import numpy as np
import struct
import math

class HandMetadata:
    __hand_parts = 21

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(HandMetadata.parse_float(four_nps[x * 4:x * 4 + 4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, img_path, ann_path, sigma):
        #self.idx = idx
        self.img = self.read_image(img_path)
        self.ann = json.load(open(ann_path, 'r'))
        self.sigma = sigma

        self.height = self.img.shape[0] #int(img_meta['height'])
        self.width = self.img.shape[1] #int(img_meta['width'])

        self.joint_list = [[(x, y, v) if v >= 1 else (-10000, -10000, 0) for x, y, v in self.ann['hand_pts']]]
        self.is_left = self.ann['is_left'] # 0 or 1

    def get_heatmap(self, target_size):
        heatmap = np.zeros((HandMetadata.__hand_parts*2, self.height, self.width), dtype=np.float32)
        shift = 0
        if self.is_left > 0:
            shift = HandMetadata.__hand_parts
        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0 or point[2] == 0:
                    continue
                HandMetadata.put_heatmap(heatmap, idx + shift, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        # heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    def get_part_mask(self):
        if self.is_left > 0:
            return np.array([0] * HandMetadata.__hand_parts + [1] * HandMetadata.__hand_parts, dtype=np.float32)
        else:
            return np.array([1] * HandMetadata.__hand_parts + [0] * HandMetadata.__hand_parts, dtype=np.float32)
    
    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y, _ = center
        _, height, width = heatmap.shape[:3]

        th = 1.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def read_image(self, img_path):
        img_str = open(img_path, "rb").read()
        if not img_str:
            print("image not read, path=%s" % img_path)
        nparr = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
