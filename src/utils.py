import logging
import os
import sys
import zipfile
import tempfile
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np

from args import get_args

try:
    from keras import backend as K
except:
    from tensorflow.keras import backend as K
from glob import glob

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt='%m/%d/%Y %I:%M:%S %p')
args = get_args()


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(logfile_name):
    try:
        if args.use_colab:
            OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}_{}'.format(args.image_size, args.prune_model,
                                                                           args.initial_sparsity,
                                                                           args.final_sparsity)
            file_handler = RotatingFileHandler('{}/logs/{}.log'.format(OUTPUT_DIR, logfile_name, mode='w'))
        else:
            file_handler = RotatingFileHandler('logs/{}.log'.format(logfile_name, mode='w'))
    except:
        raise OSError('Logs directory not created')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(logger_name))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def setup_dirs():
    try:
        if args.use_colab:
            from google.colab import drive

            drive.mount('/content/gdrive')
            OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}_{}'.format(args.image_size, args.prune_model,
                                                                           args.initial_sparsity,
                                                                           args.final_sparsity)
            if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
            log_dir = '{}/logs'.format(OUTPUT_DIR)
            if not os.path.isdir(log_dir): os.makedirs(log_dir)
        else:
            os.makedirs('logs')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def im2single(I):
    assert (I.dtype == 'uint8')
    return I.astype('float32') / 255.


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= .0).all() and (wh2 >= .0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area;
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def IOU_centre_and_dims(cc1, wh1, cc2, wh2):
    return IOU(cc1 - wh1 / 2., cc1 + wh1 / 2., cc2 - wh2 / 2., cc2 + wh2 / 2.)


def nms(Labels, iou_threshold=.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:

        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels


def image_files_from_folder(folder, upper=True):
    extensions = ['jpg', 'jpeg', 'png']
    img_files = []
    for ext in extensions:
        img_files += glob('%s/*.%s' % (folder, ext))
        if upper:
            img_files += glob('%s/*.%s' % (folder, ext.upper()))
    return img_files


def is_inside(ltest, lref):
    return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()


def crop_region(I, label, bg=0.5):
    wh = np.array(I.shape[1::-1])

    ch = I.shape[2] if len(I.shape) == 3 else 1
    tl = np.floor(label.tl() * wh).astype(int)
    br = np.ceil(label.br() * wh).astype(int)
    outwh = br - tl

    if np.prod(outwh) == 0.:
        return None

    outsize = (outwh[1], outwh[0], ch) if ch > 1 else (outwh[1], outwh[0])
    if (np.array(outsize) < 0).any():
        pause()
    Iout = np.zeros(outsize, dtype=I.dtype) + bg

    offset = np.minimum(tl, 0) * (-1)
    tl = np.maximum(tl, 0)
    br = np.minimum(br, wh)
    wh = br - tl

    Iout[offset[1]:(offset[1] + wh[1]), offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1], tl[0]:br[0]]

    return Iout


def hsv_transform(I, hsv_modifier):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I = I + hsv_modifier
    return cv2.cvtColor(I, cv2.COLOR_HSV2BGR)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= .0).all() and (wh2 >= .0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area;
    return intersection_area / union_area


def IOU_centre_and_dims(cc1, wh1, cc2, wh2):
    return IOU(cc1 - wh1 / 2., cc1 + wh1 / 2., cc2 - wh2 / 2., cc2 + wh2 / 2.)


def show(I, wname='Display'):
    cv2.imshow(wname, I)
    cv2.moveWindow(wname, 0, 0)
    key = cv2.waitKey(0) & 0xEFFFFF
    cv2.destroyWindow(wname)
    if key == 27:
        sys.exit()
    else:
        return key


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)
