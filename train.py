from keras.utils import np_utils
from numpy import *
from PIL import Image
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to remove cpu related warning
import theano

import h5py

# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def train(model,json_data,path_preprocess,img_rows,img_cols,size):
	
	imlist = []

	#batch_size to train
	batch_size = 5

	# number of output classes
	nb_classes = size

	# number of epochs to train
	nb_epoch = 20

	for i in xrange(size):
		img_name = json_data[i]['name']   # getting the name of image from final_dataset.json	
		imlist.append(img_name)

	# create matrix to store all flattened images
	immatrix = array([array(Image.open(path_preprocess + '/' + im2)).flatten() for im2 in imlist],'object')

	label = np.arange(size)

	data,Label =  shuffle(immatrix,label, random_state=2)
	train_data = [data,Label]

	img=immatrix[11].reshape(img_rows,img_cols)
	print (train_data[0].shape)
	print (train_data[1].shape)


	(X, y) = (train_data[0],train_data[1])


	# split X and y into training and testing sets

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
	
	# saving weights
	fname = "weights-Test-CNN.hdf5"
	model.save_weights(fname,overwrite=True)
	print "Data Saved"
	return X_test, Y_test
