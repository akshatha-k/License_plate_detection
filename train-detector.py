import os
from os.path import isfile, basename, splitext
from random import choice

import cv2
import keras
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from args import get_args
from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.sampler import augment_sample, labels2output_map
from src.utils import image_files_from_folder, get_logger

logger = get_logger(__name__)


def load_network(modelpath, input_dim):
    model = load_model(modelpath)
    input_shape = (input_dim, input_dim, 3)

    # Fixed input size for training
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)

    output_shape = tuple([s for s in outputs.shape[1:]])
    output_dim = output_shape[1]
    model_stride = input_dim / output_dim

    assert input_dim % output_dim == 0, \
        'The output resolution must be divisible by the input resolution'

    assert model_stride == 2 ** 4, \
        'Make sure your model generates a feature map with resolution ' \
        '16x smaller than the input'

    return model, model_stride, input_shape, output_shape


def process_data_item(data_item, dim, model_stride):
    XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim)
    YY = labels2output_map(llp, pts, dim, model_stride)
    return XX, YY


def batch_generator(X, Y, batch_size=1):
    indices = np.arange(len(X))
    batch = []
    while True:
        # it might be a good idea to shuffle your data before each epoch
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield X[batch], Y[batch]
                batch = []


if __name__ == '__main__':

    args = get_args()
    if args.use_colab:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}'.format(args.image_size, args.initial_sparsity,
                                                                    args.final_sparsity)
        if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        model_name = '{}/{}'.format(OUTPUT_DIR, args.model)
        model_path_final = '{}/{}_trained'.format(OUTPUT_DIR, args.model)
        tf_path_final = '{}/{}_trained'.format(OUTPUT_DIR, args.model)
        train_dir = '/content/gdrive/My Drive/lpd/train_images'

    netname = basename(args.name)
    outdir = OUTPUT_DIR
    batch_size = args.batch_size
    dim = args.image_size

    model, model_stride, xshape, yshape = load_network(model_name, dim)

    opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)

    print('Checking input directory...')
    logger.info('Checking input directory...')

    Files = image_files_from_folder(train_dir)

    Data = []
    for file in Files:
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            L = readShapes(labfile)
            I = cv2.imread(file)
            Data.append([I, L[0]])

    print('%d images with labels found' % len(Data))
    logger.info('%d images with labels found' % len(Data))
    # creates pool size number of datapoints from existing datapoints
    # using pre-defined augmentations.
    X, Y = [], []
    for i in range(args.num_augs):
        datapoint = choice(Data)
        x, y = process_data_item(datapoint, dim, model_stride)
        X.append(x)
        Y.append(y)

    x = np.array(X)
    y = np.array(Y)
    train_generator = batch_generator(x, y, batch_size=args.batch_size)
    callbacks = []

    if args.prune_model:
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=args.initial_sparsity, final_sparsity=args.final_sparsity,
            begin_step=args.begin_step, end_step=args.end_step)
        model = tfmot.sparsity.keras.prune_low_magnitude(
            model, pruning_schedule=pruning_schedule)

    model.compile(loss=loss, optimizer=opt)
    model.fit_generator(train_generator,
                        steps_per_epoch=(x.shape[0] // args.batch_size),
                        epochs=args.epochs,
                        callbacks=callbacks)
    print('Stopping data generator')
    logger.info('Stopping data generator')
    print('Saving model (%s)' % model_path_final)
    logger.info('Saving model (%s)' % model_path_final)
    save_model(model, model_path_final)
    tf.saved_model.save(model, tf_path_final)
