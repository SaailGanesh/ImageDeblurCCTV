#SVMusing HOG
import warnings
warnings.filterwarnings("ignore")
from os import walk
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from skimage.io import imread
import numpy as np
from segmentation.Segmentation import segment_license_plate, segment_characters
from recognition.cnn.test import classify_characters

def findli(img_pattern):
    print("INSIDE FINDLI")
    print("INSIDE FINDLI")
    print("INSIDE FINDLI")
    print(img_pattern)
    testing_dir="./static/detection_result"
    #print('Analyzing file %s with license plate' % (filename))
    img = imread('%s/%s' % (testing_dir, img_pattern))
    
    result_file = open("results.txt", "w")
    np_img = np.array(img)

    # Crop license plate in detected segment
    license_plate = segment_license_plate(np_img)
    print(type(license_plate))
    # Segment characters in detected license plate frame
    characters = segment_characters(license_plate)
    print(type(characters))
    if len(characters) != 0:
        print('Found a complete license plate: %s' % ''.join(classify_characters(characters)))
        result_file.write(''.join(classify_characters(characters)))
        s=''.join(classify_characters(characters))
        return s
    else:
        print('Unable to find all characters!')
        s="Unable to find all characters!"
        result_file.write(s)
        return s
    result_file.write("\n")
    result_file.close()

def main(img_pattern: str):
    x=findli(img_pattern[1])
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print(x)
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")
    print("OUTPUT")

if __name__ == '__main__':
    tf.compat.v1.app.run()