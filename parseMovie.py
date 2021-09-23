import math
import numpy as np
import av
import av.datasets
from skimage import io
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

#Function to take an image name and output a 128 color reduced palette and image
def ParseImg(filename, savename):
    #File path to get file
    path = './' + filename

    #Read in an array of the image file
    im = io.imread(path)

    #Get image width, height, and channel depth of image
    height, width, channel = tuple(im.shape)

    #Rearrange into 2D array with each row as a pixel
    iarr = np.reshape(im, (width * height, channel))

    #Take a random subsample of pixels in image (size 5000)
    #Run KMeans Clustering on this sample
    sample = shuffle(iarr, random_state=0)[0:5000]
    km = KMeans(n_clusters=128, random_state=0).fit(sample)

    #Assign labels of generated centroids to each pixel in original image
    labels = km.predict(iarr)
    centers = km.cluster_centers_

    #Create a new np array to store color reduced image
    reducedImage = np.zeros((height, width, channel))
    x = 0
    for i in range(height):
        for j in range(width):
            reducedImage[i][j] = centers[labels[x]]
            x = x + 1

    #Save color reduced image to new jpg
    io.imsave(savename, (reducedImage).astype(np.uint8))

    #Save clusters array to a file for parsing
    arrFile = savename.rsplit('.', 1)[0] + 'Cluster'
    arrFile2 = savename.rsplit('.', 1)[0] + 'Label'
    np.save(arrFile, centers)
    np.save(arrFile2, labels)

    return centers

#Function to downscale an image given dimensions and factor
def downscale(image, w, h, factor):
    nw = math.ceil(w/factor)
    nh = math.ceil(h/factor)
    dimg = np.zeros((nh, nw, 3), dtype=np.uint8)

    #Iterate over downscaled image size
    for i in range(nh):
        for j in range(nw):

            #Calculate the sum of rgb values binned for one downscaled pixel
            sample = np.array([0,0,0])
            count = 0
            for x in range(factor):
                for y in range(factor):
                    r = i * factor + x
                    c = j * factor + y
                    if r < w and c < h:
                        sample += image[r, c]
                        count += 1

            #Divide to get an average value
            dimg[i,j] = sample/count

    return dimg  

#Helper function to check if rgb exists in np array with specified error threshold
#RGB is passed as a float
#Returns the index of palette where it can be placed
def findPalette(array, rgb, error):
    len, _ = tuple(array.shape)
    a = array.astype(np.float32)
    b = rgb.astype(np.float32)

    bestIn = -1
    lowestErr = 100

    for i in range(len):
        temp = float(0)
        temp += abs(a[i][0] - b[0])
        temp += abs(a[i][1] - b[1])
        temp += abs(a[i][2] - b[2])

        #Iterate through palette and find best match
        if temp < error:
            if temp < lowestErr:
                bestIn = i
                lowestErr = temp

    return bestIn, lowestErr

#Helper function to check if rgb exists in np array with specified error threshold
#RGB is passed as a float
#Returns the index of palette where it can be placed
#This is an edited version of the original function that only checks indexes in unfilledPalette
def findPaletteB(array, rgb, error, indexes):
    a = array.astype(np.float32)
    b = rgb.astype(np.float32)

    bestIn = -1
    lowestErr = 100

    for i in range(len(indexes)):
        x = indexes[i]
        temp = float(0)
        temp += abs(a[x][0] - b[0])
        temp += abs(a[x][1] - b[1])
        temp += abs(a[x][2] - b[2])

        #Iterate through palette and find best match
        if temp < error:
            if temp < lowestErr:
                bestIn = x
                lowestErr = temp

    return bestIn, lowestErr

#Helper function to calculate the average RGB value of an image
def averageRGB(image, w, h):
    r = float(0)
    g = float(0)
    b = float(0)

    #Sum squares of all rgb values
    for i in range(w):
        for j in range(h):
            r += float(image[i][j][0]) * float(image[i][j][0])
            g += float(image[i][j][1]) * float(image[i][j][1])
            b += float(image[i][j][2]) * float(image[i][j][2])

    #Take the sqrt of average rgb values squared
    r = math.sqrt(r/(w*h))
    g = math.sqrt(g/(w*h))
    b = math.sqrt(b/(w*h))

    result = np.array([r,g,b])
    return result

#Parses a video file and fills a palette with downscaled images
def PaletteFrames(videoname, foldername, p):
    #Open the video as a container and parse key frames
    with av.open(videoname) as container:
        output = container.streams.video[0]
        #Skip non keyframes
        output.codec_context.skip_frame = 'NONKEY'

        #Count number of palette colors filled
        created = np.zeros(128)
        filled = 0

        for frame in container.decode(output):

            #Convert frame to a np array and extract shape
            img = frame.to_image()
            arr = np.asarray(img)
            height, width, channel = tuple(arr.shape)

            #Downscale the key frame image
            dm = downscale(arr, width, height, 45)
            w, h, c = tuple(dm.shape)

            #Get the average RGB from the key frame
            avg = averageRGB(dm, w, h)

            #Check if frames satisfies a palette
            num, err = findPalette(palette, avg, 20)

            #Match found
            if num != -1:
                #Create a palette jpg if it hasn't been created before
                if created[num] == 0:
                    print("Added", num)
                    created[num] = err
                    filled += 1

                    name = './' + foldername + '/' + str(num) + '.jpg'
                    io.imsave(name, dm, check_contrast=False)
                #Replace a palette jpg if it has lower error value
                elif err < created[num]:
                    print("Replace", num)
                    created[num] = err

                    name = './' + foldername + '/' + str(num) + '.jpg'
                    io.imsave(name, dm, check_contrast=False)

            if filled >= 128:
                print("Palette filled")
                break

        print("First iteration through movie keyframes complete.")

        #Check if all palette colors have been added
        unfilledPalette = []

        for i in range(128):
            if created[i] == 0:
                unfilledPalette.append(i)

        return unfilledPalette

#Function to generate jpgs for remaining empty palettes
def FinishPalettes(videoname, foldername, palette, error, emptyPalette):
    #Open the video as a container and parse key frames
    with av.open(videoname) as container:
        output = container.streams.video[0]
        #Skip non keyframes
        output.codec_context.skip_frame = 'NONKEY'

        #Count number of palette colors filled
        created = np.zeros(128)
        filled = 0

        for frame in container.decode(output):
            #Convert frame to a np array and extract shape
            img = frame.to_image()
            arr = np.asarray(img)
            height, width, channel = tuple(arr.shape)

            #Downscale the key frame image
            dm = downscale(arr, width, height, 45)
            w, h, c = tuple(dm.shape)

            #Get the average RGB from the key frame
            avg = averageRGB(dm, w, h)

            #Check if frames satisfies a palette
            num, err = findPaletteB(palette, avg, error, emptyPalette)

            #Match found
            if num != -1:
                #Create a palette jpg if it hasn't been created before
                if created[num] == 0:
                    print("Added", num)
                    created[num] = err
                    filled += 1

                    name = './' + foldername + '/' + str(num) + '.jpg'
                    io.imsave(name, dm, check_contrast=False)
                #Replace a palette jpg if it has lower error value
                elif err < created[num]:
                    print("Replace", num)
                    created[num] = err

                    name = './' + foldername + '/' + str(num) + '.jpg'
                    io.imsave(name, dm, check_contrast=False)

            if filled >= len(emptyPalette):
                print("Palette filled")
                break

        unfilledPalette = []

        for i in range(len(emptyPalette)):
            x = emptyPalette[i]
            if created[x] == 0:
                unfilledPalette.append(x)

        return unfilledPalette

#Function calls to parse image, create a palette, and fill a folder with downscale jpgs
#Filled in with sample file names and directories
def main():
    #Parse Image and generate a K Means Palette
    ParseImg('strange.jpg', 'strangeReduced.jpg')

    #Parse the movie by key frames and save which palette colors haven't been saved
    temp = PaletteFrames('Strange.mp4', 'Strange', palette)
    print("Palettes not filled:")
    print(temp)

    #Load palette from file
    palette = np.load('strangeReducedCluster.npy')

    #Reiterate through movie with higher error threshold
    arr = FinishPalettes('Strange.mp4', 'Strange', palette, 30, temp)
    print(arr)

#Commented out since new main call generates a slightly different palette
#main()
