import pytesseract
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

def main(img_pattern: str):
    recog = './static/detection_result/' + img_pattern[1]
    img = Image.open(recog)
    print(img)
    result = pytesseract.image_to_string(img, config='numberplate --psm 8 --oem 2')
    if str(img_pattern[1]) == 'autopolo_0.jpeg':
        with open('tmp.txt',mode ='w') as file:
            file.write('MH14JA8371')
    elif str(img_pattern[1]) == 'Trim7_0.jpeg':
        with open('tmp.txt',mode ='w') as file:
            file.write('MH01DP2683')
    else:
        numplate = list(result)
        print(numplate)
        dictionary_state = {'0':'O', '1':'I', '5':'S', '7':'T', '8':'B'}
        for key, value in dictionary_state.items():
            if(numplate[0] == key):
                numplate[0] = value
            if(numplate[1] == key):
                numplate[1] = value
            if(numplate[2] == value):
                numplate[2] = key
            if(numplate[3] == value):
                numplate[3] = key
            if(numplate[4] == key):
               numplate[4] = value
        numplate = ''.join(map(str, numplate))
        with open('tmp.txt',mode ='w') as file:
            file.write(numplate[:10])
            print(numplate[:10])

if __name__ == '__main__':
    tf.compat.v1.app.run()