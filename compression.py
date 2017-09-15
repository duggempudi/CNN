import numpy
import random
from PIL import Image
def compress(image):
	newimage=[]
	for i in range(0,28,2):
		newrow=[]
		for j in range(0,28,2):
			newrow.append((image[i][j]+image[i][j+1]+image[i+1][j]+image[i+1][j+1])/4)
		newimage.append(newrow)
	newimage=numpy.asarray(newimage)
	return newimage
def compress2(image):
	newimage=[]
	for i in range(0,len(image)-1):
		newrow=[]
		for j in range(0,len(image)-1):
			newrow.append((image[i][j]+image[i][j+1]+image[i+1][j]+image[i+1][j+1])/4)
		newimage.append(newrow)
	newimage=numpy.asarray(newimage)
	return newimage	
def compressbatch(images):
	functions=[minimize,minimize1,minimize2,minimize3]
	compressedimages=[]					
	for i in range(len(images)):
		rand=functions[random.randint(0,3)]
		compressedimages.append(uncompress(changeimage(modify(images[i]))))
	numpy.asarray(compressedimages)
	return compressedimages	
def modify(image):
	newimage=[]
	for i in range(28):
		newimage.append(image[28*i:28*(i+1)])
	newimage=numpy.asarray(newimage)
	return newimage
def uncompress(image):
	newimage=[]
	for i in image:
		for j in i:
			newimage.append(j)
	return newimage
def minimize(image):
	newimage=[]
	for i in range(28):
		row=[]
		for j in range(28):
			if i>=14 and j<14:
				row.append(image[i-14][j])
			else:
				row.append(0)
		newimage.append(row)
	return newimage
def minimize1(image):
	newimage=[]
	for i in range(28):
		row=[]
		for j in range(28):
			if i<14 and j<14:
				row.append(image[i][j])
			else:
				row.append(0)
		newimage.append(row)
	return newimage
def minimize2(image):
	newimage=[]
	for i in range(28):
		row=[]
		for j in range(28):
			if i>=14 and j>=14:
				row.append(image[i-14][j-14])
			else:
				row.append(0)
		newimage.append(row)
	return newimage
def minimize3(image):
	newimage=[]
	for i in range(28):
		row=[]
		for j in range(28):
			if i<14 and j>=14:
				row.append(image[i][j-14])
			else:
				row.append(0)
		newimage.append(row)
	return newimage	
def printimage(image):
	for i in image:
		for j in i:
			if j>0:
				print '*',
			else:
				print ' ',
		print
def expandbatch(image):
	newimage=[]
	for i in range(28):
		row1=[]
		row2=[]
		for j in range(28):
			row1.append(image[i][j])
			row1.append(image[i][j])
			row2.append(image[i][j])
			row2.append(image[i][j])
		newimage.append(row1)
		newimage.append(row2)
	return numpy.asarray(newimage)		
def changeimage(image):
	rand=random.randint(6,28)
	x=random.randint(0,28-rand)
	y=random.randint(0,28-rand)
	img=numpy.asarray(Image.fromarray(numpy.uint8(image*256)).resize((rand,rand))).astype(float)/256
	newimage=numpy.zeros((28,28),dtype=float)
	newimage[x:x+rand,y:y+rand]=img
	return newimage	
def reduceimage(image):
	return numpy.asarray(Image.fromarray(numpy.uint8(image*256)).resize((20,20))).astype(float)/256		
def reducebatch(images):
	compressedimages=[]					
	for i in range(len(images)):
		compressedimages.append(uncompress(reduceimage(modify(images[i]))))
	numpy.asarray(compressedimages)
	return compressedimages

			
