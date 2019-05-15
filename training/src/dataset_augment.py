# -*- coding: utf-8 -*-
# @Time    : 18-3-7 下午2:36
# @Author  : edvard_hua@live.com
# @FileName: dataset_augument.py
# @Software: PyCharm

import math
import random

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid
from enum import Enum

_network_w = 256
_network_h = 256
_scale = 2


class CocoPart(Enum):
    Top = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Background = 14


def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h


def set_network_scale(scale):
    global _scale
    _scale = scale


def get_network_output_wh():
    return _network_w // _scale, _network_h // _scale


def pose_crop_portrait(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta
    # crop top
    h = meta.height
    for joint in meta.joint_list:
        Top = joint[CocoPart.Top.value]
        LWrist = joint[CocoPart.LWrist.value]
        RWrist = joint[CocoPart.RWrist.value]
        min_y = min(LWrist[1] if LWrist[2]>0 else h, RWrist[1] if RWrist[2]>0 else h)
        min_y = min(min_y, Top[1] if Top[2]>0 else h)
        max_y = max(0, min(min_y, h-_network_h))
    crop_top = random.randint(0, max_y)
    # crop bottom
    min_y = meta.height
    max_y = 0
    for joint in meta.joint_list:
        LHip = joint[CocoPart.LHip.value]
        RHip = joint[CocoPart.RHip.value]
        LWrist = joint[CocoPart.LWrist.value]
        RWrist = joint[CocoPart.RWrist.value]
        min_y = min(LHip[1] if LHip[2]>0 else min_y, RHip[1] if RHip[2]>0 else min_y)
        max_y = max(LWrist[1] if LWrist[2]>0 else max_y, RWrist[1] if RWrist[2]>0 else max_y)
        break
    if min_y < _network_h + crop_top:
        min_y = min(_network_h + crop_top, meta.height)
    crop_bottom = random.randint(int(max(min_y, max_y)), meta.height)
    return pose_crop(meta, 0, crop_top, meta.width, crop_bottom - crop_top)

def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)

    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            adjust_joint.append((point[0] * (neww/float(meta.width)), point[1] * (newh/float(meta.height)), point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta

def hand_rotation(meta):
    deg = random.uniform(-15.0, 15.0)
    img = meta.img
    (h, w) = img.shape[:2]
    center = (w * 0.5, h * 0.5)  # x, y
    rot_m = cv2.getRotationMatrix2D((center[0], center[1]), deg, 1)
    cos = np.abs(rot_m[0,0])
    sin = np.abs(rot_m[0,1])
    neww = int((h*sin) + (w*cos) + 0.5)
    newh = int((h*cos) + (w*sin) + 0.5)
    rot_m[0,2] += (neww/2) - center[0]
    rot_m[1,2] += (newh/2) - center[1]
    ret = cv2.warpAffine(img, rot_m, (neww, newh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    newx, newy = 0, 0
    img = ret

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                adjust_joint.append((-10000, -10000, 0))
                continue
            #x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point[:2], deg)
            x, y = point[:2]
            xx = rot_m[0,0] * x + rot_m[0,1] * y + rot_m[0,2]
            yy = rot_m[1,0] * x + rot_m[1,1] * y + rot_m[1,2]
            adjust_joint.append((xx, yy, point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = img.shape[1], img.shape[0]
    meta.img = img

    return meta

def pose_rotation(meta):
    deg = random.uniform(-15.0, 15.0)
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)  # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    # print(ret.shape, deg, newx, newy, neww, newh)
    img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                adjust_joint.append((-10000, -10000, 0))
                continue
            x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point[:2], deg)
            adjust_joint.append((x, y, point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = img

    return meta


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    flip_list = [CocoPart.Top, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder,
                 CocoPart.RElbow, CocoPart.RWrist,
                 CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle
                 ]
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for cocopart in flip_list:
            point = joint[cocopart.value]
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                adjust_joint.append((-10000, -10000, 0))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((meta.width - 1 - point[0], point[1], point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_resize_shortestedge_random(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.8, 1.2)) # 0.95 -> 0.8
    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))
    return pose_resize_shortestedge(meta, target_size)


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return qx, qy


def pose_resize_shortestedge(meta, target_size):
    global _network_w, _network_h
    img = meta.img
    if not target_size:
        target_size = min(_network_h, _network_w)
    # adjust image
    scale = target_size / min(meta.height, meta.width)
    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    pw = ph = 0
    if neww < _network_w or newh < _network_h:
        pw = max(0, (_network_w - neww) // 2)
        ph = max(0, (_network_h - newh) // 2)
        mw = (_network_w - neww) % 2
        mh = (_network_h - newh) % 2
        color1 = random.randint(0, 255)
        color2 = random.randint(0, 255)
        color3 = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(color1, color2, color3))

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                adjust_joint.append((-10000, -10000, 0))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((point[0] * (neww/float(meta.width)) + pw, point[1] * (newh/float(meta.height)) + ph, point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww + pw * 2, newh + ph * 2
    meta.img = dst
    return meta


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y + target_size[1], x:x + target_size[0], :]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            new_x, new_y, new_v = point[0] - x, point[1] - y, point[2]
            adjust_joint.append((new_x, new_y, new_v))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = target_size
    meta.img = resized
    return meta


def pose_crop_random(meta):
    global _network_w, _network_h
    if (meta.width == _network_w and meta.height == _network_h):
        return meta
    target_size = (_network_w, _network_h)
    x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
    y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0
    #for _ in range(50):
    #    x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
    #    y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0
    #
    #    # check whether any face is inside the box to generate a reasonably-balanced datasets
    #    for joint in meta.joint_list:
    #        if      x <= joint[CocoPart.RKnee.value][0] < x + target_size[0] and \
    #                y <= joint[CocoPart.RKnee.value][1] < y + target_size[1] and \
    #                x <= joint[CocoPart.RAnkle.value][0] < x + target_size[0] and \
    #                y <= joint[CocoPart.RAnkle.value][1] < y + target_size[1] and \
    #                x <= joint[CocoPart.LKnee.value][0] < x + target_size[0] and \
    #                y <= joint[CocoPart.LKnee.value][1] < y + target_size[1] and \
    #                x <= joint[CocoPart.LAnkle.value][0] < x + target_size[0] and \
    #                y <= joint[CocoPart.LAnkle.value][1] < y + target_size[1]:
    #            break
    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    return meta_l.img.astype(np.float32), \
           meta_l.get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)).astype(np.float32)


def hand_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                adjust_joint.append((-10000, -10000, 0))
                continue
            adjust_joint.append((meta.width - 1 - point[0], point[1], point[2]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.img = img
    meta.is_left = 1 - meta.is_left # exchange left and right
    return meta

def hand_crop(meta):
    left = meta.width - 1
    right = 0
    top = meta.height - 1
    bottom = 0
    for joint in meta.joint_list:
        for point in joint:
            if point[0] < -1000 or point[1] < -1000 or point[2] == 0:
                continue
            left = min(left, point[0])
            right = max(right, point[0])
            top = min(top, point[1])
            bottom = max(bottom, point[1])
    width = right - left + 1
    height = bottom - top + 1
    
    # expand half
    left -= max((height - width)/2, width / 2)
    right += max((height - width)/2, width / 2)
    top -= max((width - height)/2, height / 2)
    bottom += max((width - height)/2, height / 2)

    # expand randomly
    max_expand = max(width, height) * 2.5
    left  -= min(max_expand, abs(np.random.normal(0, width / 2)))
    right += min(max_expand, abs(np.random.normal(0, width / 2)))
    top   -= min(max_expand, abs(np.random.normal(0, height / 2)))
    bottom+= min(max_expand, abs(np.random.normal(0, height / 2)))

    # legal int
    left = int(max(0, round(left)))
    right = int(min(meta.width - 1, round(right)))
    top = int(max(0, round(top)))
    bottom = int(min(meta.height - 1, round(bottom)))

    # expand to necessary pixels
    min_pixels = min(64, meta.width, meta.height)
    to_expand = max(bottom - top + 1, min_pixels, meta.width)
    if right - left + 1 < to_expand:
        pad = (to_expand-1 - (right - left)) // 2
        left = max(0, left - pad)
        pad = to_expand-1 - (right - left)
        right = min(meta.width - 1, right + pad)
        pad = to_expand-1 - (right - left)
        left = max(0, left - pad)
    to_expand = max(right - left + 1, min_pixels)
    if bottom - top + 1 < to_expand:
        pad = (to_expand-1 - (bottom - top)) // 2
        top = max(0, top - pad)
        pad = to_expand-1 - (bottom - top)
        bottom = min(meta.height - 1, bottom + pad)
        pad = to_expand-1 - (bottom - top)
        top = max(0, top - pad)
        
    return pose_crop(meta, int(left), int(top), int(right - left + 1), int(bottom - top + 1))

def hand_to_img(meta_l):
    global _network_w, _network_h, _scale
    return meta_l.img.astype(np.float32), \
           meta_l.get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)).astype(np.float32), \
           meta_l.get_part_mask()
