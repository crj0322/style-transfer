import tensorflow as tf
import numpy as np
from keras.applications import vgg16
from keras.layers import Input, Lambda, Conv2D, ReLU, UpSampling2D
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K

from style_swap import StyleSwap

class FastStyleSwap(StyleSwap):
    def __init__(self, input_size=[512, 512, 3], 
        base_model_fn=vgg16.VGG16, preprocess=vgg16.preprocess_input, 
        out_layer_name='block3_conv1', patch_size=3, 
        style_batch=1, content_batch=1, 
        inverse_net=None):
        """
        Args:
            input_size(list of int): input image size.
            base_model_fn(function): model function used for build base model.
            preprocess(function): preprocess function like: (lambda x: (x/127.5. - 1.)).
            out_layer_name(str): output layer name of base model for style swap.
            patch_size(int): size of style patch.
            style_batch(int): count of input style images.
            content_batch(int): count of input content images.
            inverse_net(str): h5 model file path.
        """
        if inverse_net != None:
            self.inverse_net = load_model(inverse_net)
        else:
            self._build_inverse_net()
        
        super().__init__(input_size=input_size, 
            base_model_fn=base_model_fn, preprocess=preprocess, 
            out_layer_name=out_layer_name, patch_size=patch_size, 
            style_batch=style_batch, content_batch=content_batch)
    
    def _build_swap_model(self):
        super()._build_swap_model()
        generated_img = self.inverse_net(self.swap_model.outputs[0])
        self.evaluate_op = K.function([*self.swap_model.inputs, K.learning_phase()], [generated_img])

    def _build_inverse_net(self):
        inputs = Input([self._input_size[0]//4, self._input_size[1]//4, 256])

        def instance_norm(inputs):
            return tf.contrib.layers.instance_norm(inputs)
        
        # assume x as shape[None, w/4, h/4, 256]
        x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(inputs)
        x = Lambda(instance_norm)(x)
        x = ReLU()(x)

        # upsample to [w/2, h/2]
        x = UpSampling2D()(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
        x = Lambda(instance_norm)(x)
        x = ReLU()(x)

        x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
        x = Lambda(instance_norm)(x)
        x = ReLU()(x)

        # upsample to [w, h]
        x = UpSampling2D()(x)
        x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
        x = Lambda(instance_norm)(x)
        x = ReLU()(x)

        x = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu')(x)

        self.inverse_net = Model(inputs, x)
    
    def evaluate(self, style, content):
        """
        generate transfered image.

        Args:
            style(numpy array): input style rgb image.
            content(numpy array): input content rgb image.

        Returns:
            numpy array: rgb transfered image of uint8.
        """
        style = np.expand_dims(style, axis=0)
        content = np.expand_dims(content, axis=0)
        img = self.evaluate_op([style, content, 0])[0]
        img = self.depreprocess(img)
        return img
        
    def compile_inverse_net(self, lr=1e-2, tv_reg=0.5):
        """
        build training model.

        Args:
            lr(float): learning rate.
            tv_reg(float): image continuous regularization term.

        Returns:
            keras model for training.
        """
        swaped_feature = self.inverse_net.inputs[0]
        generated_img = self.inverse_net.outputs[0]
        generated_feature = self.base_model(generated_img)
        loss = Lambda(self._calc_loss, arguments={'tv_reg': tv_reg})\
            ([swaped_feature, generated_feature, generated_img])
        
        train_model = Model(swaped_feature, loss)
        opt = optimizers.Adam(lr=lr)
        train_model.compile(opt, loss=(lambda y_true, y_pred: y_pred))
        return train_model


def main():
    fast_swap = FastStyleSwap(inverse_net='deep_inverse_512.h5')
    
    import cv2 as cv
    from utils import squar_resize
    style = cv.imread('style.jpg')
    content = cv.imread('content.jpg')
    style = squar_resize(style, 512)
    content = squar_resize(content, 512)
    cv.imshow('style', style)
    cv.waitKey()
    cv.imshow('content', content)
    cv.waitKey()
    style = style[:, :, ::-1]
    content = content[:, :, ::-1]

    img = fast_swap.evaluate(style, content)
    img = img[:, :, ::-1]
    cv.imshow('transfered', img)
    cv.waitKey()

    cv.destroyAllWindows()
    

if __name__ == '__main__':
    main()