
import sys
import numpy as np
import pandas as pd
import cv2
import argparse
import keras
import tensorflow_model_optimization as tfmot
from args import get_args
from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator,CustomDataGenerator

from pdb import set_trace as pause


def load_network(modelpath,input_dim, prune_model):

	model = load_model(modelpath)
	if prune_model:
		model = tfmot.sparsity.keras.prune_low_magnitude(model)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple([s for s in outputs.shape[1:]])
	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

def process_data_item(data_item,dim,model_stride):
	XX,llp,pts = augment_sample(data_item[0],data_item[1].pts,dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY


if __name__ == '__main__':

	args= get_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	model,model_stride,xshape,yshape = load_network(args.model,dim,args.prune_model)

	opt = getattr(keras.optimizers,args.optimizer)(lr=args.learning_rate)
	model.compile(loss=loss, optimizer=opt)

	print('Checking input directory...')
	Files = image_files_from_folder(train_dir)

	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			I = cv2.imread(file)
			Data.append([I,L[0]])

	print('%d images with labels found' % len(Data))

	df=pd.DataFrame(Data, columns=['image','labels'])
	# dg = CustomDataGenerator(	data=Data, \
	# 					process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
	# 					xshape=xshape, \
	# 					yshape=(yshape[0],yshape[1],yshape[2]+1), \
	# 					nthreads=2, \
	# 					pool_size=1000, \
	# 					min_nsamples=100 )
	dg = CustomDataGenerator(df=df,xshape = xshape, yshape= (yshape[0],yshape[1],yshape[2]+1))
	# dg.start()

	Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
	Ytrain = np.empty((batch_size,int(dim//model_stride),int(dim//model_stride),2*4+1))

	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)
	callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
	# for it in range(iterations):

	# 	print('Iter. %d (of %d)' % (it+1,iterations))

	# 	Xtrain,Ytrain = dg.get_batch(batch_size)
	# 	train_loss = model.train_on_batch(Xtrain,Ytrain,callbacks=callbacks)

	# 	print('\tLoss: %f' % train_loss)

	# 	# Save model every 1000 iterations
	# 	if (it+1) % 1000 == 0:
	# 		print('Saving model (%s)' % model_path_backup)
	# 		save_model(model,model_path_backup)
	model.fit_generator(model, dg, epochs=1)
	print('Stopping data generator')
	dg.stop()

	print('Saving model (%s)' % model_path_final)
	save_model(model,model_path_final)
