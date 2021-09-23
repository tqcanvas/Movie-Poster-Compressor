import math
import numpy as np
from skimage import io

#Get a filepath name given index
def genPath(folderName, index):
	path = './' + folderName + '/' + str(index) + '.jpg'

	return path

#Iterate over numpy array of palette indices and build a complete image
def buildDitheredImage(array, folderName, filename, origH, origW):
	array = array.astype(np.uint8)
	height, width = tuple(array.shape)

	bigH = origH * 9
	bigW = origW * 16

	bigPic = np.zeros((bigH, bigW, 3), dtype=np.uint8)

	for y in range(height):
		print("Starting row:", y)
		for x in range(width):
			ind = array[y][x]

			path = genPath(folderName, ind)
			img = io.imread(path).astype(np.uint8)

			for i in range(9):
				for j in range(16):
					bigPic[y * 9 + i][x * 16 + j] = img[i][j]

	io.imsave(filename, bigPic)

#Main function call
def main():
	arr = np.load('strangeDither.npy')
	buildDitheredImage(arr, 'Strange', 'strangeFinal.jpg', 2048, 1382)

#main()
