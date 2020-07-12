import tensorflow as tf
from tensorflow.keras.layers import (Concatenate, Conv2D, Dropout, Input,
                                     MaxPooling2D, Softmax, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def GetOutChannel(depth):
    """ 
    """
    assert (depth >= 0 and depth < 5)
    channels = [32, 64, 128, 256, 512]
    return channels[depth]

def EncodeBlock(inputs, out_channel, train=True):
    conv = Conv2D(out_channel, (3, 3), activation='relu', padding='same')(inputs)
    if train:
        conv = Dropout(0.2)(conv)
    conv = Conv2D(out_channel, (3, 3), activation='relu', padding='same')(conv)
    pool = MaxPooling2D((2, 2))(conv)
    return conv, pool

def DecodeBlock(inputs, skip_conv, out_channel, train=True):
    up = UpSampling2D(size=(2, 2))(inputs)
    concat = Concatenate(axis=3)([skip_conv, up])
    conv = Conv2D(out_channel, (1, 1), activation='relu', padding='same')(concat)
    if train:
        conv = Dropout(0.2)(conv)
    conv = Conv2D(out_channel, (3, 3), activation='relu', padding='same')(conv)
    if train:
        conv = Dropout(0.2)(conv)
    conv = Conv2D(out_channel, (3, 3), activation='relu', padding='same')(conv)
    return conv

def Unet(shape, encode_depth, train=True):
    """
    shape: shape of inputs, don't include batch size.
    encode_depth: the number of encode blocks, same as decode blocks.
    """

    # Inputs.
    inputs = Input(shape=shape)
    
    # Encode.
    conv_snapshots = []
    flow = inputs
    for i in range(encode_depth):
        out_channel = GetOutChannel(i)
        conv, flow = EncodeBlock(flow, out_channel, train=train)
        conv_snapshots.append(conv)

    flow = conv_snapshots[encode_depth-1]

    # Decode.
    for i in range(encode_depth-2, -1, -1):
        out_channel = GetOutChannel(i)
        flow = DecodeBlock(flow, conv_snapshots[i], out_channel, train=train)

    # Outputs.
    outputs = Conv2D(2, (1, 1), activation='relu', padding='same')(flow)
    outputs = Softmax()(outputs)
    
    # Compile model.
    model = Model(inputs=inputs, outputs=outputs)
    opt = Adam()
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    unet = Unet((512, 512, 1), 5)
    plot_model(
        unet,
        show_shapes=True,
        show_layer_names=False,
        to_file='./logs/test_model.png')