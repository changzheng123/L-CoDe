from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# General options
__C.IMG_DIR = ''
__C.MODEL_DIR = ''
__C.MODEL_FOLDER = ''
__C.PRE_MODEL_DIR = ''
__C.RESULT_DIR = ''
__C.RESULT_FOLDER = ''
__C.RESOURCE_DIR = 'resources/'
__C.LOG_FILE = ''
__C.TEST_INFO = ''
__C.TEST_DIR = ''
__C.CAT2ID = ''
__C.SINGULAR = ''
__C.WORD2INDEX = ''
__C.TRAIN_CAPTION = ''
__C.VAL_CAPTION = ''
__C.DRAW_ATTN = False
__C.GPU_group = [0]
__C.WORKERS = 0
__C.RNN_TYPE = ''

# Training options
__C.TRAIN = edict()
__C.TRAIN.CKPT = -1
__C.TRAIN.HEIGHT = 224
__C.TRAIN.WIDTH = 224
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.MAX_EPOCH = 12
__C.TRAIN.SAVE_INTERVAL = 1
__C.TRAIN.LEARNING_RATE = 0.00002
__C.TRAIN.PRINT_FREQUENCY = 100
__C.TRAIN.EVAL_FREQUENCY = 3
__C.TRAIN.LINEAR = False
__C.TRAIN.VGGL = 10.0
__C.TRAIN.GTTHRESH = 0.5
__C.TRAIN.ENCODER_LR = 0.00002
__C.TRAIN.USEMASK = True
__C.TRAIN.OBJ2COL = True

# Pretrain options
__C.PRETRAIN = edict()
__C.PRETRAIN.SMOOTH = edict()
__C.PRETRAIN.SMOOTH.GAMMA1 = 1.0
__C.PRETRAIN.SMOOTH.GAMMA2 = 1.0
__C.PRETRAIN.SMOOTH.GAMMA3 = 1.0
__C.PRETRAIN.SMOOTH.LAMBDA = 1.0
__C.PRETRAIN.CKPT = 0

# Text options
__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 300
__C.TEXT.CAPTIONS_PER_IMAGE = 1
__C.TEXT.RANDOM = 0.2
__C.TEXT.WORDS_NUM = 8
__C.TEXT.SEPARATOR_COLOR = '+'
__C.TEXT.SEPARATOR_NUM = '-'

# Data generator
__C.DATA = edict()
__C.DATA.INFO = ''
__C.DATA.SEGMENTATION = ''
__C.DATA.GROUNDTRUTH = ''
__C.DATA.RESULT_TXT = ''
__C.DATA.RESULT_JSON = ''
__C.DATA.THRESHOLD_COLOR = 0.35
__C.DATA.THRESHOLD_PROPORTION = 0.1
__C.DATA.THRESHOLD_GRAYSCALE = 0.8
__C.DATA.GRAYSCALE = ['black', 'gray', 'white']
__C.DATA.DIVISION = {'black': [[[0, 0, 0], [180, 255, 39]]],
                     'gray': [[[0, 13, 40], [180, 26, 255]],
                              [[0, 0, 40], [180, 12, 209]]],
                     'white': [[[0, 0, 210], [180, 12, 255]]],
                     'red': [[[0, 164, 40], [7, 255, 255]],
                             [[170, 164, 40], [180, 255, 255]]],
                     'orange': [[[8, 27, 40], [20, 255, 255]]],
                     'yellow': [[[21, 27, 40], [30, 255, 255]]],
                     'green': [[[31, 27, 40], [78, 255, 255]],
                               [[79, 154, 40], [85, 255, 255]]],
                     'blue': [[[79, 27, 40], [85, 153, 255]],
                              [[86, 27, 40], [123, 255, 255]],
                              [[124, 201, 40], [131, 255, 255]]],
                     'purple': [[[124, 27, 40], [131, 200, 255]],
                                [[132, 27, 40], [143, 255, 255]]],
                     'pink': [[[0, 27, 40], [7, 163, 255]],
                              [[144, 27, 40], [170, 255, 255]],
                              [[170, 27, 40], [180, 163, 255]]]}
__C.DATA.ADJUST_DELETE = ['skis', 'sports ball', 'baseball bat', 'baseball glove', 'fork',
                          'spoon', 'toaster', 'hair drier', 'parking meter', 'traffic light', 'tennis racket']
__C.DATA.ADJUST_RENAME = {'door-stuff': 'door', 'mirror-stuff': 'mirror', 'water-other': 'water',
                          'tree-merged': 'tree', 'fence-merged': 'fence', 'ceiling-merged': 'ceiling',
                          'sky-other-merged': 'sky', 'cabinet-merged': 'cabinet', 'table-merged': 'table',
                          'pavement-merged': 'pavement', 'mountain-merged': 'mountain', 'grass-merged': 'grass',
                          'dirt-merged': 'dirt', 'paper-merged': 'paper', 'building-other-merged': 'building',
                          'rock-merged': 'rock', 'rug-merged': 'rug', 'food-other-merged': 'food',
                          'fire hydrant': 'fire-hydrant', 'stop sign': 'sign',
                          'hot dog': 'hot-dog', 'potted plant': 'plant',
                          'cell phone': 'cell-phone', 'teddy bear': 'teddy-bear',
                          'wine glass': 'glass'}
__C.DATA.ADJUST_MERGE = {'wall': ['wall-other-merged', 'wall-wood', 'wall-brick', 'wall-stone', 'wall-tile'],
                         'window': ['window-blind', 'window-other'],
                         'floor': ['floor-wood', 'floor-other-merged'],
                         'table': ['dining table', 'table']}



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
