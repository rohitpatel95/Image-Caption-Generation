import sys
from PIL import Image 
import visualize as vl

path = sys.argv[1]
img = Image.open(path)   
resized_img = img.resize((100,100))	    	
gray_img = resized_img.convert('L')   

gray_img.save("input_img.jpg", "JPEG") 
file = open("sen.txt","w");
sent = vl.visualize() 
file.write(sent)
file.close()
print sent
