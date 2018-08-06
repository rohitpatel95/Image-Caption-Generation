import h5py
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to remove cpu related warning
from PIL import Image
import numpy as np
import json

def visualize():

	# load json and create model
	path_json = '/root/keras3/final_dataset.json'
	json_data = []	
	#loading json dataset     
	with open(path_json) as data_file:            
		json_data = json.load(data_file)
	json_file = open('model.json', 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)

	# loading Weights
	fname = "weights-Test-CNN.hdf5"
	model.load_weights(fname)

	img = Image.open("input_img.jpg")
	rimg = np.array(img).reshape(1, 100,100,1)
	class_id = model.predict_classes(rimg)
	img_sentence = json_data[class_id[0]]["sentence"]
	return img_sentence
