from IPython.display import Image, display
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Some example images
filenames = glob.glob("images/*.jpg")
filenames.sort()

for img in filenames:
    img = mpimg.imread(img)
    imgplot = plt.imshow(img)
    plt.show()
