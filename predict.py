import h5py

def predict(model,json_data,X_test,Y_test,size):
	
	# Loading weights

	fname = "weights-Test-CNN.hdf5"
	model.load_weights(fname)
	print "Image Loaded"

	#------------------   

	mat_predict = model.predict_classes(X_test)
	mat_original = Y_test
	len_predict = len(mat_predict)


	print "\nPredicted Sentences :\n"
	for i in range(5):
		img_id = mat_predict[i]
		img_sentence = json_data[img_id]["sentence"]	
		print img_sentence
	print "\nActual Sentences :\n"
	for i in range(5):
		for j in range(size):
			if Y_test[i][j] == 1:
				img_id = j
				img_sentence = json_data[img_id]["sentence"]			
				print img_sentence

	#------------------

	print mat_predict
	print mat_original

