from os import makedirs
from os.path import isfile, isdir, basename, splitext
from random import choice

import cv2
import keras
import numpy as np
import tensorflow_model_optimization as tfmot

from args import get_args
from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.sampler import augment_sample, labels2output_map
from src.utils import image_files_from_folder


def load_network(modelpath, input_dim, prune_model):
  model = load_model(modelpath)
  # if prune_model:
  #	model = tfmot.sparsity.keras.prune_low_magnitude(model)
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

  netname = basename(args.name)
  train_dir = args.train_dir
  outdir = args.output_dir

  iterations = args.iterations
  batch_size = args.batch_size
  dim = 208

  if not isdir(outdir):
    makedirs(outdir)

  model, model_stride, xshape, yshape = load_network(args.model, dim, args.prune_model)

  opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)

  print('Checking input directory...')
  Files = image_files_from_folder(train_dir)

  Data = []
  for file in Files:
    labfile = splitext(file)[0] + '.txt'
    if isfile(labfile):
      L = readShapes(labfile)
      I = cv2.imread(file)
      Data.append([I, L[0]])

  print('%d images with labels found' % len(Data))

  # creates pool size number of datapoints from existing datapoints
  # using pre-defined augmentations.
  X, Y = [], []
  for i in range(1000):
    datapoint = choice(Data)
    x, y = process_data_item(datapoint, dim, model_stride)
    X.append(x)
    Y.append(y)

  x = np.array(X)
  y = np.array(Y)
  train_generator = batch_generator(x, y, batch_size=args.batch_size)

  # dg = CustomDataGenerator(	data=Data, \
  # 					process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
  # 					xshape=xshape, \
  # 					yshape=(yshape[0],yshape[1],yshape[2]+1), \
  # 					nthreads=2, \
  # 					pool_size=1000, \
  # 					min_nsamples=100 )
  # dg = CustomDataGenerator(df=df,xshape = xshape, yshape= (yshape[0],yshape[1],yshape[2]+1))
  # dg.start()

  # Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
  # Ytrain = np.empty((batch_size,int(dim//model_stride),int(dim//model_stride),2*4+1))

  model_path_backup = '%s/%s_backup' % (outdir, netname)
  model_path_final = '%s/%s_final' % (outdir, netname)
  callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
  pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0, final_sparsity=0.5,
    begin_step=0, end_step=200)
  model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)
  model_for_pruning.compile(loss=loss, optimizer=opt)

  # for it in range(iterations):

  # 	print('Iter. %d (of %d)' % (it+1,iterations))

  # 	Xtrain,Ytrain = dg.get_batch(batch_size)
  # 	train_loss = model.train_on_batch(Xtrain,Ytrain,callbacks=callbacks)

  # 	print('\tLoss: %f' % train_loss)

  # 	# Save model every 1000 iterations
  # 	if (it+1) % 1000 == 0:
  # 		print('Saving model (%s)' % model_path_backup)
  # 		save_model(model,model_path_backup)
  model_for_pruning.fit_generator(train_generator,
                      steps_per_epoch=(x.shape[0] // args.batch_size),
                      epochs=1,
                      callbacks=callbacks)
  print('Stopping data generator')
  # dg.stop()

  print('Saving model (%s)' % model_path_final)
  save_model(model, model_path_final)
