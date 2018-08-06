import pre_process as pp
import train as train
import predict as predict

#KERAS
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

def extractFeature():
	json_data,path_preprocess,img_rows,img_cols,size = pp.preProcess()
	
	# number of convolutional filters to use
	nb_filters = 32

	# size of pooling area for max pooling
	nb_pool = 2

	# convolution kernel size
	nb_conv = 3

	# number of output classes
	nb_classes = size

	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv) , padding='valid',input_shape=( img_rows, img_cols,1)))
	
	convout1 = Activation('relu')
	model.add(convout1)
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
	
	convout2 = Activation('relu')
	model.add(convout2)
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')

	X_test, Y_test = train.train(model,json_data,path_preprocess,img_rows,img_cols,size)
	predict.predict(model,json_data,X_test,Y_test,size)
	
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	print "json model saved"

if __name__ == "__main__":
	extractFeature()

