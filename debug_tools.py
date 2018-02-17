import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Takes numpy column/row array of image and plots figure
#
#   x = Numpy Array
#   rows = Number of rows
#   cols = Number of columns

def plotImg(x, rows, cols):
    reshapedX = np.reshape(x, (rows, cols))
    imgplot = plt.imshow(reshapedX, cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(10)
    print("finished plot")
    return