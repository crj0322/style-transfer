import tensorflow as tf
import cv2 as cv
import time
from fast_style_swap import FastStyleSwap
from utils import squar_resize

flags = tf.app.flags

flags.DEFINE_string('inverse_net', './deep_inverse_512.h5', 'Path to inverse net model file.')
flags.DEFINE_string('content', './data/content.jpg', 'Path to content image.')
flags.DEFINE_string('style', './data/style.jpg', 'Path to style image.')
flags.DEFINE_string('output', './data/output.jpg', 'Path to transfered image.')
flags.DEFINE_integer('size', 512, 'Net input image size')
FLAGS = flags.FLAGS

def main(_):
    fast_swap = FastStyleSwap(input_size=[FLAGS.size, FLAGS.size, 3], 
        inverse_net=FLAGS.inverse_net)
    
    style = cv.imread(FLAGS.style)
    content = cv.imread(FLAGS.content)
    if style.shape[0] > FLAGS.size or style.shape[1] > FLAGS.size:
        style = squar_resize(style, FLAGS.size)
    else:
        # padding
        pass
    if content.shape[0] > FLAGS.size or content.shape[1] > FLAGS.size:
        content = squar_resize(content, FLAGS.size)
    else:
        # padding
        pass
    cv.imshow('style', style)
    # cv.waitKey()
    cv.imshow('content', content)
    cv.waitKey()
    style = style[:, :, ::-1]
    content = content[:, :, ::-1]

    start = time.clock()
    img = fast_swap.predict(style, content)
    end = time.clock()
    print('run time: %.2f seconds'%(end-start))
    img = img[:, :, ::-1]
    cv.imshow('transfered', img)
    cv.waitKey()
 
    cv.destroyAllWindows()

    cv.imwrite(FLAGS.output, img)

if __name__ == '__main__':
    tf.app.run()