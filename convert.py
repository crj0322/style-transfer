import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, Lambda
from keras.models import Model

def reflectpadding(x):
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    return x

def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = Lambda(reflectpadding)(x)
    x = Conv2D(filters, kernel_size, activation=activation, padding='valid')(x)
    return x

def build_model():
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

    model = Model(inputs, x)
    return model

def get_weights(reader, layer_name):
    kernel = reader.get_tensor(layer_name + '/kernel')
    bias = reader.get_tensor(layer_name + '/bias')
    return [kernel, bias]


# from https://github.com/elleryqueenhomels/arbitrary_style_transfer
cpktFileName = r'./arbitrary_style_model_style-weight/models/style_weight_2e0.ckpt'
reader = tf.train.NewCheckpointReader(cpktFileName)
# for key in sorted(reader.get_variable_to_shape_map()):
#     print(key)

layer_name = ['decoder/conv4_1', 
    'decoder/conv3_4', 'decoder/conv3_3', 'decoder/conv3_2', 'decoder/conv3_1', 
    'decoder/conv2_2', 'decoder/conv2_1', 
    'decoder/conv1_2', 'decoder/conv1_1']

model = build_model()
i = 0
for layer in model.layers:
    if layer.name.startswith('conv2d'):
        weights = get_weights(reader, layer_name[i])
        layer.set_weights(weights)
        i += 1

model.save_weights('inverse_net.h5')