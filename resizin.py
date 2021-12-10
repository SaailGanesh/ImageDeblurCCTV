from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
idir='./static/resize_img'
op_dir='./static/selectedframes'
def main(img_pattern: str):
    image = Image.open('%s/%s' % (idir, img_pattern[1]))
    print(img_pattern[1])
    width,height=image.size
    print(width,height)
    if width>height:
        new_image = image.resize((1280, 720))
    else:
        new_image = image.resize((720, 1280))
    new_image.save('%s/%s'% (op_dir, img_pattern[1]))
    
if __name__ == '__main__':
    tf.compat.v1.app.run()