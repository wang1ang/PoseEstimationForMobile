import tensorflow as tf
from dataset_augment import pose_random_scale, pose_rotation, pose_resize_shortestedge_random, pose_crop_random, hand_crop, hand_flip, hand_to_img, pose_resize_shortestedge, hand_rotation
from dataset_hand_prepare import HandMetadata
#from os.path import join
from os import listdir
#import multiprocessing
import json
BASE = "/root/hdd"
BASE_PATH = 'F:\\upperbody\\cmuHand\\'
CONFIG = None
def set_config(config):
    global CONFIG, BASE, BASE_PATH
    CONFIG = config
    BASE = CONFIG['imgpath']
    BASE_PATH = CONFIG['datapath']

def load_sample(filename):
    #print (filename)
    #print (type(filename))
    sizes = []
    image = filename.decode() + ".jpg"
    anno = filename.decode() + ".json"
    meta_data = HandMetadata(image, anno, sigma=6.0)
    sizes.append([0, meta_data.width, meta_data.height])
    meta_data = pose_random_scale(meta_data) # 1
    sizes.append([1, meta_data.width, meta_data.height])
    meta_data = hand_rotation(meta_data) # 2
    sizes.append([2, meta_data.width, meta_data.height])
    meta_data = hand_crop(meta_data) # 3
    sizes.append([3, meta_data.width, meta_data.height])
    meta_data = pose_resize_shortestedge(meta_data, None) # 4
    sizes.append([4, meta_data.width, meta_data.height])
    if meta_data.height != 192 or meta_data.width != 192 or meta_data.img.shape[0] != 192 or meta_data.img.shape[1] != 192 or meta_data.img.shape[2] != 3:
        print (sizes)
        print (meta_data.img.shape)
    meta_data = hand_flip(meta_data) # 5
    meta_data = pose_crop_random(meta_data)
    return hand_to_img(meta_data)
def get_test_files(root):
    manual = root + 'hand_labels\\manual_test\\'
    manual = [manual + f[:-5] for f in listdir(manual) if f.endswith('.json')]
    return manual
def get_train_files(root):
    manual = root + 'hand_labels\\manual_train\\'
    manual = [manual + f[:-5] for f in listdir(manual) if f.endswith('.json')]
    synth = []
    for i in ['synth1\\', 'synth2\\', 'synth3\\', 'synth4\\']:
        synth = synth + [root + 'hand_labels_synth\\' + i + f[:-5] for f in listdir(root + 'hand_labels_synth\\' + i) if f.endswith('.json')]
    return manual # + synth
    #return manual
def get_dataset(batch_size=32, epoch=10, buffer_size=20, is_train=True):
    files = get_train_files(BASE_PATH) if is_train else get_test_files(BASE_PATH)
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(
        lambda filename: tuple(
            tf.py_func(
                func=load_sample,
                inp=[filename],
                Tout=[tf.float32, tf.float32, tf.float32]
            )
        ), num_parallel_calls=CONFIG['multiprocessing_num']
    )
    dataset = dataset.map(_set_shapes, num_parallel_calls=CONFIG['multiprocessing_num'])
    dataset = dataset.batch(batch_size).repeat(epoch)
    dataset = dataset.prefetch(100)
    return dataset

def _set_shapes(img, heatmap, mask):
    img.set_shape([CONFIG['input_height'], CONFIG['input_width'], 3])
    heatmap.set_shape([CONFIG['input_height'] / CONFIG['scale'], CONFIG['input_width'] / CONFIG['scale'], CONFIG['n_kpoints']])
    mask.set_shape(CONFIG['n_kpoints'])
    return img, heatmap, mask

#files = get_train_files(BASE_PATH)
#print (len(files))
#f = files[0]
#sample = load_sample(f.encode('utf-8'))
#print (sample[0].shape)
#print (sample[1].shape)
#print (sample[2].shape)

