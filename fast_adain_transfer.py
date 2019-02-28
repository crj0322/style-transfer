import tensorflow as tf
import numpy as np
from keras.applications import vgg19
from keras.layers import Input, Lambda, Conv2D, ReLU, UpSampling2D
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K

from adain_transfer import AdainTransfer

class FastAdainTransfer(AdainTransfer):
    def __init__(self, base_model_fn=vgg19.VGG19, preprocess=vgg19.preprocess_input, 
        out_layer_name=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], 
        inverse_weight=None):
        """
        Args:
            base_model_fn(function): model function used for build base model.
            preprocess(function): preprocess function like: (lambda x: (x/127.5. - 1.)).
            out_layer_name(str): output layer name of base model for calculation.
            inverse_weight(str): h5 model file path.
        """
        self.inverse_net = self._build_inverse_net()
        if inverse_weight != None:
            self.inverse_net.load_weights(inverse_weight)
        
        super().__init__(base_model_fn=base_model_fn, preprocess=preprocess, 
            out_layer_name=out_layer_name)
    
    def _build_transfer_model(self, base_model_fn, preprocess, out_layer_name):
        super()._build_transfer_model(base_model_fn, preprocess, out_layer_name)
        generated_img = self.inverse_net(self.encoder.outputs[-1])
        self.predict_model = Model(self.encoder.inputs, generated_img)

    def _build_inverse_net(self):
        def reflectpadding(x):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            return x

        def conv_block(x, filters, kernel_size=3, activation='relu'):
            x = Lambda(reflectpadding)(x)
            x = Conv2D(filters, kernel_size, activation=activation, padding='valid')(x)
            return x

        inputs = Input(shape=[None, None, 512])
        x = conv_block(inputs, 256)
        x = UpSampling2D()(x)

        x = conv_block(x, 256)
        x = conv_block(x, 256)
        x = conv_block(x, 256)
        x = conv_block(x, 128)
        x = UpSampling2D()(x)

        x = conv_block(x, 128)
        x = conv_block(x, 64)
        x = UpSampling2D()(x)

        x = conv_block(x, 64)
        x = conv_block(x, 3, activation=None)

        return Model(inputs, x)
    
    def predict(self, style, content):
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
        img = self.predict_model.predict([content, style])
        img = self.roundimg(img)
        return img
        
    def compile_inverse_net(self, lr=1e-2, lambd=0.5):
        """
        build training model.

        Args:
            lr(float): learning rate.
            lambd(float): content style trade-off.

        Returns:
            keras model for training.
        """
        phi_s_t = self.encoder.outputs
        generated_img = self.inverse_net(phi_s_t[-1])
        phi_gt = self.base_model(generated_img)
        loss = Lambda(self._calc_loss, arguments={'lambd': lambd})\
            ([*phi_s_t, *phi_gt])
        
        train_model = Model(phi_s_t, loss)
        opt = optimizers.Adam(lr=lr)
        train_model.compile(opt, loss=(lambda y_true, y_pred: y_pred))
        return train_model
