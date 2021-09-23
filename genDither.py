import math
import numpy as np
from skimage import io

#Function to read original image and return a 2D numpy array using Floyd-Steinberg Dithering
#to recreate the image with 128 palette colors
def fsDither(imageName, paletteFileName):
	#Load in original image
	path = './' + imageName
	img = io.imread(path)

	rows, cols, d = tuple(img.shape)

	img = img.astype(np.float32)

	#Load in palette file
	palette = np.load(paletteFileName)


	#Create array to store output and errors
	res = np.zeros((rows,cols))

	#Floyd Steinberg Algorithm
	for y in range(rows):
		for x in range(cols):
			realP = img[y][x]
			minIndex, error = closest(realP, palette)
			res[y][x] = minIndex

			if x+1 < cols:
				img[y][x+1] += error * 7/16
			if y+1 < rows:
				if x > 0:
					img[y+1][x-1] += error * 3/16
				img[y+1][x] += error * 5/16
			if x+1 < cols and y+1 < rows:
				img[y+1][x+1] += error * 1/16

	return res

#Helper function to find closest palette color given color
def closest(color, palette):
	#Index of closest palette color
	minI = 0

	#Calculate absolute difference
	diff = np.sum(np.absolute(color - palette[0]))

	for i in range(128):
		dist = np.sum(np.absolute(color - palette[i]))

		if (dist < diff):
			minI = i
			diff = dist

	return minI, np.array(color - palette[minI])

#Main function to create a dithered array of palette indices
def main():
	print("Building an Floyd-Steinberg Dithering Array of Palette indexes:")
	arr = fsDither('strange.jpg', 'strangeReducedCluster.npy')
	np.save('strangeDither', arr)
	print("The dither array shape is:")
	print(tuple(arr.shape))

#main()
