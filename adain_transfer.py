import tensorflow as tf
import numpy as np
from keras.applications import vgg19
from keras.layers import Input, Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K


class AdainTransfer():
    def __init__(self, base_model_fn=vgg19.VGG19, preprocess=vgg19.preprocess_input, 
        out_layer_name=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']):
        """
        Args:
            base_model_fn(function): model function used for build base model.
            preprocess(function): preprocess function like: (lambda x: (x/127.5. - 1.)).
            out_layer_name(str): output layer name of base model for calculation.
        """

        self._build_transfer_model(base_model_fn, preprocess, out_layer_name)

    def _build_transfer_model(self, base_model_fn, preprocess, out_layer_name):
        # pretrained model
        base_inputs = Input(shape=[None, None, 3])
        base_model = base_model_fn(input_tensor=Lambda(preprocess)(base_inputs), 
            include_top=False)
        outputs = [base_model.get_layer(layer_name).output for layer_name in out_layer_name]
        base_model = Model(base_inputs, outputs)
        for layer in base_model.layers:
            layer.trainable = False
        self.base_model = base_model

        # get feature map
        input_content = Input(shape=[None, None, 3], name='content')
        input_style = Input(shape=[None, None, 3], name='style')
        content_feature = base_model(input_content)[-1]
        style_feature_list = base_model(input_style)

        # adaptive instance normalization
        transfered_feature = Lambda(self._adain)([content_feature, style_feature_list[-1]])
        
        self.encoder = Model([input_content, input_style], [*style_feature_list, transfered_feature])

    def _moment(self, tensor, axis=[1, 2]):
        mu = K.mean(tensor, axis=axis)
        var = K.var(tensor, axis=axis)
        sigma = K.sqrt(var + K.epsilon())
        return mu, sigma
    
    def _adain(self, args):
        content, style = args
        mu_c, sigma_c = self._moment(content)
        mu_s, sigma_s = self._moment(style)

        return sigma_s * (content - mu_c) / sigma_c + mu_s

    def _calc_loss(self, args, lambd):
        phi_s = args[:4]
        t = args[4]
        phi_gt = args[5:]

        def mse(a, b):
            return K.mean(K.square(a - b))
        
        L_c = mse(phi_gt[-1], t)
        L_s = 0
        for s, gt in zip(phi_s, phi_gt):
            mu_gt, sigma_gt = self._moment(gt)
            mu_s, sigma_s = self._moment(s)
            L_s += mse(mu_gt, mu_s) + mse(sigma_gt, sigma_s)

        loss = L_c + lambd * L_s / len(phi_s)
        return loss

    def roundimg(self, img):
        """
        transform output values to rgb image.

        Args:
            img(numpy array): output image of model.
            
        Returns:
            numpy array: rgb image of uint8.
        """
        img = np.squeeze(img)
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def build_train_fn(self, input_size, lr=1e-1, lambd=0.5, init_img=None):
        """
        build keras train function.

        Args:
            lr(float): learning rate.
            lambd(float): content style trade-off.
            init_img(numpy array): used for initialize generated image.
        """
        # loss
        phi_s_t = [Input(l.get_shape().as_list()[1:]) for l in self.encoder.outputs]
        if init_img == None:
            init_img = np.random.uniform(low=-0.1, high=0.1, size=[1, *input_size])
        self.generated_img = K.variable(init_img)
        phi_gt = self.base_model(self.generated_img)
        loss = self._calc_loss([*phi_s_t, *phi_gt], lambd)
        
        # Define optimizer and training function
        self.lr = lr
        self.optimizer = optimizers.Adam(lr=lr)
        updates_op = self.optimizer.get_updates(params=[self.generated_img], loss=loss)
        self.train_fn = K.function(
            inputs=[*phi_s_t, K.learning_phase()],
            outputs=[loss],
            updates=updates_op)

    def train_image(self, style, content, epochs, print_interval=10, patience=5):
        """
        train transfered image.

        Args:
            style(numpy array): input style rgb image.
            content(numpy array): input content rgb image.
            epochs(int): training epochs.
            print_interval(int): print loss every x epochs.
            patience(int): if loss do not decrease for x epochs, than multiply lr by 0.1 .
        """
        style = np.expand_dims(style, axis=0)
        content = np.expand_dims(content, axis=0)
        inputs = self.encoder.predict([content, style])

        min_loss = np.inf
        min_count = 0
        lr = self.lr
        for i in range(epochs):
            loss = self.train_fn([*inputs, 1])[0]
            if loss < min_loss:
                min_loss = loss
                min_count = i
            elif i - min_count > patience:
                min_count = i
                lr *= 0.1
                K.set_value(self.optimizer.lr, lr)
                print('current lr: %.4f' % (lr))
            if lr <= 1e-4:
                break

            if i % print_interval == 0:
                print('epoch %d: loss = %.4f' % (i, loss))

        print('final loss = %.4f' % (loss))
        img = K.get_value(self.generated_img)
        
        return self.roundimg(img)
