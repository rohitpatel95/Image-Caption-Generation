import json
from PIL import Image

def preProcess():
	path_raw = '/root/keras3/raw_dataset'
	path_preprocess = '/root/keras3/preprocessed_dataset'
	path_json = '/root/keras3/final_dataset.json'

	json_data = []
	
	#loading json dataset     
	with open(path_json) as data_file:            
		json_data = json.load(data_file)

	# dimension of resized image
	img_rows = 100
	img_cols = 100

	size = 2500   # number of images in dataset
	print size
	
	for i in xrange(size):
		img_name = json_data[i]['name']      # getting the name of image from flickr8k.json
		img = Image.open(path_raw + '/' + img_name)     # loading image
		resized_img = img.resize((img_rows,img_cols))
	    	gray_img = resized_img.convert('L')       # converting resized image into gray scale          
	    	gray_img.save(path_preprocess +'/' + img_name, "JPEG")   
		if i%50 == 0:
			print i  
	print "Pre Process Done"
	return json_data,path_preprocess,img_rows,img_cols,size
