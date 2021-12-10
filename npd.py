# Required libraries
import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode
from absl import app, flags
#from lpr import findli
from absl.flags import FLAGS
import cv2
import numpy as np
import time

import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # limits gpu memory usage

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline() # downloads pretrained weights for text detector and recognizer

tf.keras.backend.clear_session()

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

flags.DEFINE_string('input', 'inputs/1.jpg', 'path to input image')
flags.DEFINE_string('output', 'results/output.jpg', 'path to save results')
flags.DEFINE_integer('size', 608, 'resize images to')

def platePattern(string):
    '''Returns true if passed string follows
    the pattern of indian license plates,
    returns false otherwise.
    '''
    if len(string) < 9 or len(string) > 10:
        return False
    elif string[:2].isalpha() == False:
        return False
    elif string[2].isnumeric() == False:
        return False
    elif string[-4:].isnumeric() == False:
        return False
    elif string[-6:-4].isalpha() == False:
        return False
    else:
        return True
    
def drawText(img, plates):
    '''Draws recognized plate numbers on the
    top-left side of frame
    '''
    string  = 'plates detected :- ' + plates[0]
    for i in range(1, len(plates)):
        string = string + ', ' + plates[i]
    
    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    (text_width, text_height) = cv2.getTextSize(string, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((1, 30), (10 + text_width, 20 - text_height))
    print(box_coords)
    cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(img, string, (5, 25), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
    
def plateDetect(frame, input_size, model):
    '''Preprocesses image and pass it to
    trained model for license plate detection.
    Returns bounding box coordinates.
    '''
    frame_size = frame.shape[:2]
    image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    
    return bboxes




# Match contours to license plate or character template
def _check_contours(boundaries, img_orig, img_preproc, license_plate_check) :

    # Find all contours in the image
    (cntrs, _) = cv2.findContours(img_preproc.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential boundaries
    lower_width = boundaries[0]
    upper_width = boundaries[1]
    lower_height = boundaries[2]
    upper_height = boundaries[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    if license_plate_check is True :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:5]
    else :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        # Check if contour has proper sizes
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :

            x_cntr_list.append(intX)
            target_contours.append(cntr)

            # If we check a license plate, crop the license plate
            if license_plate_check is True :
                img_res = img_orig[intY:intY+intHeight, intX:intX+intWidth, :]

            # If we check a character, crop the character
            if license_plate_check is False :

                char_copy = np.zeros((44,24))
                char = img_orig[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy)

    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
    if license_plate_check is not True:
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        target_contours_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])
            target_contours_copy.append(target_contours[idx])
        img_res = img_res_copy
        target_contours = target_contours_copy

    return target_contours, img_res




'''

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (150, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,147:150] = 255

    # Estimations of character contours sizes of cropped license plates
    boundaries_crop = [LP_WIDTH/6,
                       LP_WIDTH/3,
                       LP_HEIGHT/6,
                       2*LP_HEIGHT/3]

    # Estimations of character contour sizes of non-cropped license plates
    boundaries_no_crop = [LP_WIDTH/12,
                          LP_WIDTH/6,
                          LP_HEIGHT/8,
                          LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_contours, char_list = _check_contours(boundaries_crop, img_binary_lp, img_binary_lp, False)

    if len(char_contours) != 7:

        # Check the smaller contours; possibly no plate was detected at all
        char_contours, char_list = _check_contours(boundaries_no_crop, img_binary_lp, img_binary_lp, False)


    if len(char_contours) == 0 :

            # If nothing was found, try inverting the image in case the background is darker than the foreground
            invert_img_lp = np.invert(img_binary_lp)
            char_contours, char_list = _check_contours(boundaries_crop, img_binary_lp, invert_img_lp, False)

    # If we found 7 chars, it is likely to form a license plate
    full_license_plate = []
    full_license_plate = char_list

    return full_license_plate
'''

'''
def lilicen(image):
    x=findli(image)
    print("\n\n\n\n")
    print(x)
    print("\n\n\n\n")
'''







def main(_argv):
    
    input_layer = tf.keras.layers.Input([FLAGS.size, FLAGS.size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, 'data/YOLOv4-obj_1000.weights')

    img = cv2.imread(FLAGS.input) # Reading input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = plateDetect(img, FLAGS.size, model) # License plate detection
    plates = []
    #xywh 
    #0123
    #0   1   2   3
    #237 128 570 292
    print("LEN")
    print("LEN")
    print("LEN")

    for i in range(len(bboxes)):
        plate_img = img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])]
        print(int(bboxes[i][1]),int(bboxes[i][3]), int(bboxes[i][0]),int(bboxes[i][2]))
        crop2_img=img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])]
        #cv2.imwrite("crop2.jpg", crop2_img)
    cv2.imwrite(FLAGS.output, crop2_img)

'''
        print("newcode")
        ymin = bboxes[i][0]
        xmin = bboxes[i][1]
        ymax = bboxes[i][2]
        xmax = bboxes[i][3]
        (im_width, im_height) = image.size
        (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
        print("Plate detected: ", (xminn, xmaxx, yminn, ymaxx))


        prediction_groups = pipeline.recognize([plate_img]) # Text detection and recognition on license plate
        string = ''
        for j in range(len(prediction_groups[0])):
            string = string+ prediction_groups[0][j][0].upper()

        if platePattern(string) == True and string not in plates:
            plates.append(string)

    if len(plates) > 0:
        drawText(img, plates)

    cv2.imwrite(FLAGS.output, crop2_img) # Saving output
    print('Output saved to ', FLAGS.output)
'''


    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
