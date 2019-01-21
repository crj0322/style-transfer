import tensorflow as tf
import numpy as np
from keras.applications import vgg16
from keras.layers import Input, Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K


class StyleSwap():
    def __init__(self, input_size=[224, 224, 3], 
        base_model_fn=vgg16.VGG16, preprocess=vgg16.preprocess_input, 
        out_layer_name='block3_conv1', patch_size=3, 
        style_batch=1, content_batch=1):
        """
        Args:
            input_size(list of int): input image size.
            base_model_fn(function): model function used for build base model.
            preprocess(function): preprocess function like: (lambda x: (x/127.5. - 1.)).
            out_layer_name(str): output layer name of base model for style swap.
            patch_size(int): size of style patch.
            style_batch(int): count of input style images.
            content_batch(int): count of input content images.
        """
        self._input_size = input_size
        self._base_model_fn = base_model_fn
        self._preprocess = preprocess
        self._out_layer_name = out_layer_name
        self._patch_size = patch_size
        self._style_batch = style_batch
        self._content_batch = content_batch

        self._build_swap_model()

    def _build_swap_model(self):
        # pretrained model
        base_inputs = Input(shape=self._input_size)
        preprocess_layer = Lambda(self._preprocess)
        base_model = self._base_model_fn(input_tensor=preprocess_layer(base_inputs), 
            input_shape=self._input_size, include_top=False)
        base_model = Model(base_inputs, base_model.get_layer(self._out_layer_name).output)
        for layer in base_model.layers:
            layer.trainable = False
        self.base_model = base_model

        # get feature map
        input_content = Input(batch_shape=(self._content_batch, *self._input_size), name='content')
        input_style = Input(batch_shape=(self._style_batch, *self._input_size), name='style')
        content_feature = base_model(input_content)
        style_feature = base_model(input_style)

        # fast swap
        swaped_feature = Lambda(self._fast_swap)([style_feature, content_feature])
        self.swap_model = Model([input_style, input_content], swaped_feature)

    def _get_style_patches(self, style):
        style_shape = style.get_shape().as_list()
        f = self._patch_size
        h = style_shape[1]
        w = style_shape[2]

        # split extracted feature in filter size for row
        rows = tf.split(style, num_or_size_splits=list(
                [f] * (h // f) + [w % f]), axis=1)[:-1]
        # split every row in filter size for colum
        cells = [tf.split(row, num_or_size_splits=list(
                [f] * (h // f) + [w % f]), axis=2)[:-1]
                for row in rows]
        # collect all patch
        stacked_cells = [tf.stack(row_cell, axis=4) for row_cell in cells]
        style_patches = tf.concat(stacked_cells, axis=-1)

        return style_patches
    
    def _fast_swap(self, tensor_list):
        style, content = tensor_list
        style_amount = style.get_shape()[0].value
        batch_filters = self._get_style_patches(style)
        swaped_list = []
        for filters in tf.unstack(batch_filters, axis=0, num=style_amount):
            normalized_filters = tf.nn.l2_normalize(filters, axis=(0, 1, 2))

            # dot product of each content patch and normalized style patch
            # assume content stride = 1
            similarity = tf.nn.conv2d(content, normalized_filters, 
                strides=[1, 1, 1, 1], padding="VALID")

            # replace content patch with a style patch of max similarity
            arg_max_K = tf.argmax(similarity, axis=-1)
            one_hot_K = tf.one_hot(arg_max_K, depth=similarity.get_shape()[-1].value)
            swap = tf.nn.conv2d_transpose(one_hot_K, filters, output_shape=tf.shape(content),
                strides=[1, 1, 1, 1], padding="VALID")

            # average overlapping values for stride 1
            swap /= 9.0

            swaped_list.append(swap)

        return tf.concat(swaped_list, axis=0)

    def _calc_loss(self, tensor_list, tv_reg):
        swaped_feature, generated_feature, generated_img = tensor_list
        h, w = self._input_size[:2]

        left_tensor = tf.slice(generated_img, [0, 0, 0, 0], [-1, w-1, -1, -1])
        right_tensor = tf.slice(generated_img, [0, 1, 0, 0], [-1, -1, -1, -1])
        up_tensor = tf.slice(generated_img, [0, 0, 0, 0], [-1, -1, h-1, -1])
        down_tensor = tf.slice(generated_img, [0, 0, 1, 0], [-1, -1, -1, -1])
        ltv = K.mean(K.square(left_tensor - right_tensor))/2 + \
            K.mean(K.square(up_tensor - down_tensor))/2

        loss = K.mean(K.square(generated_feature - swaped_feature))/2 + tv_reg*ltv
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

    def build_train_fn(self, lr=1e-1, tv_reg=0.5, init_img=None):
        """
        build keras train function.

        Args:
            lr(float): learning rate.
            tv_reg(float): image continuous regularization term.
            init_img(numpy array): used for initialize generated image.
        """
        # loss
        swaped_feature = Input(self.swap_model.outputs[0].get_shape().as_list()[1:])
        if init_img == None:
            init_img = np.random.uniform(low=-0.1, high=0.1, size=[1, *self._input_size])
        self.generated_img = K.variable(init_img)
        generated_feature = self.base_model(self.generated_img)
        loss = self._calc_loss([swaped_feature, generated_feature, self.generated_img], tv_reg)
        
        # Define optimizer and training function
        self.lr = lr
        self.optimizer = optimizers.Adam(lr=lr)
        updates_op = self.optimizer.get_updates(params=[self.generated_img], loss=loss)
        self.train_fn = K.function(
            inputs=[swaped_feature, K.learning_phase()],
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
        swaped_feature = self.swap_model.predict([style, content])

        min_loss = np.inf
        min_count = 0
        lr = self.lr
        for i in range(epochs):
            loss = self.train_fn([swaped_feature, 1])[0]
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
