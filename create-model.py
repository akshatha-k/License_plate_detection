import os

import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from keras.models import Model

from args import get_args
from src.keras_utils import save_model


def res_block(x, sz, filter_sz=3, in_conv_size=1):
    xi = x
    for i in range(in_conv_size):
        xi = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
        xi = BatchNormalization()(xi)
        xi = Activation('relu')(xi)
    xi = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
    xi = BatchNormalization()(xi)
    xi = Add()([xi, x])
    xi = Activation('relu')(xi)
    return xi


def conv_batch(_input, fsz, csz, activation='relu', padding='same', strides=(1, 1)):
    output = Conv2D(fsz, csz, activation='linear', padding=padding, strides=strides)(_input)
    output = BatchNormalization()(output)
    output = Activation(activation)(output)
    return output


def end_block(x):
    xprobs = Conv2D(2, 3, activation='softmax', padding='same')(x)
    xbbox = Conv2D(6, 3, activation='linear', padding='same')(x)
    return Concatenate(3)([xprobs, xbbox])


def create_model_eccv():
    input_layer = Input(shape=(None, None, 3), name='input')

    x = conv_batch(input_layer, 16, 3)
    x = conv_batch(x, 16, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 32, 3)
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 128, 3)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)

    x = end_block(x)

    return Model(inputs=input_layer, outputs=x)


# Model not converging...
def create_model_mobnet():
    input_layer = Input(shape=(None, None, 3), name='input')
    x = input_layer

    mbnet = MobileNet(input_shape=(224, 224, 3), include_top=True)

    backbone = keras.models.clone_model(mbnet)
    for i, bblayer in enumerate(backbone.layers[1:74]):
        layer = bblayer.__class__.from_config(bblayer.get_config())
        layer.name = 'backbone_' + layer.name
        x = layer(x)

    x = end_block(x)

    model = Model(inputs=input_layer, outputs=x)

    backbone_layers = {'backbone_' + layer.name: layer for layer in backbone.layers}
    for layer in model.layers:
        if layer.name in backbone_layers:
            print('setting ' + layer.name)
            layer.set_weights(backbone_layers[layer.name].get_weights())

    return model


def create_model(args):
    if args.model == 'eccv':
        model = create_model_eccv()
    else:
        model = create_model_mobnet()
    return model


if __name__ == '__main__':
    args = get_args()
    if args.use_colab:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}'.format(args.image_size, args.initial_sparsity,
                                                                    args.final_sparsity)
        if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        model_name = '{}/{}'.format(OUTPUT_DIR, args.model)

    assert (args.model == 'eccv' or args.model == 'mobnet'), 'Model name must be on of the following: eccv or mobnet'

    model = create_model(args)

    print('Saving at %s' % model_name)
    save_model(model, model_name)
