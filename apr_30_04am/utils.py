import numpy as np
from PIL import Image


""" convert the input image from a color 256*240 pixel to 84* 84 black and white image without the scoreboard partern"""
def convertImage(img):
    img = Image.fromarray(img)
    img = img.convert("L")
    img = img.resize((84, 110), Image.BILINEAR)
    img = np.array(img)
    img = img[18:102, :]
    return img.astype(np.uint8)

def filePreflixProcession(isDueling=False, isDropout=False, dropoutRate=0.0):
    filePreflix = ""
    if isDueling:
        filePreflix += "dueling_"
    if isDropout:
        filePreflix += "dropout" + str(dropoutRate) + "_"
    return filePreflix
