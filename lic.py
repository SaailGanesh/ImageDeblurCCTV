import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
path = './static/white/white.jpg'
image = cv2.imread(path)
window_name = 'Image'
f=open('tmp.txt',"r")
x=f.readlines()
text = str(x[0])
font = cv2.FONT_HERSHEY_SIMPLEX
org = (00, 185)
fontScale = 1
color = (0, 0,0)
thickness = 2
image = cv2.putText(image, text, org, font, fontScale,color, thickness, cv2.LINE_AA)
op_path = './static/Output/autopolo_0.jpeg'
cv2.imwrite(op_path,image)
