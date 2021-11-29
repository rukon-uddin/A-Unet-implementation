import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, add, UpSampling2D, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def convBlock(input, num_filters):
    x = Conv2D(num_filters, 3, activation='relu', padding='same')(input)
    x = Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    return x


def gating(input, output_size):
    x = Conv2D(output_size, 1, padding='same')(input)
    x = Activation('relu')(x)
    return x


def attention_gate(gating, x, shape):
    theta_x = Conv2D(shape, (1, 1), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
    phi_g = Conv2D(shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)

    concat_xg = add([theta_x, phi_g])
    act_xg = Activation('relu')(concat_xg)

    psi = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)

    upsampling_psi = UpSampling2D(size=(2, 2))(sigmoid_xg)

    y = multiply([upsampling_psi, x])
    # shape_x = K.int_shape(y)
    # print("y x = ", shape_x)

    result = Conv2D(3, 1, padding='same')(y)
    att = BatchNormalization()(result)
    return att


def build_attention_unet(input):

    inputs = Input(input)
    x = convBlock(inputs, 64)
    f1 = convBlock(inputs, 64)  # 512,512,3
    x = MaxPool2D(2, 2)(x)

    x = convBlock(x, 128)
    f2 = convBlock(x, 128)  # 256,256,3
    x = MaxPool2D(2, 2)(x)

    x = convBlock(x, 256)
    f3 = convBlock(x, 256)  # 128,128,3
    x = MaxPool2D(2, 2)(x)

    f4 = convBlock(x, 512)  # 64,64,3

    g4 = gating(f4, 256)
    a4 = attention_gate(g4, f3, 256)
    up_conv_f4 = UpSampling2D(size=(2, 2))(f4)
    concat_4 = Concatenate([up_conv_f4, a4])

    f2_up = convBlock(up_conv_f4, 256)
    g3 = gating(f2_up, 128)
    a3 = attention_gate(g3, f2, 128)
    up_conv_f2_up = UpSampling2D(size=(2, 2))(f2_up)
    concat_3 = Concatenate([a3, up_conv_f2_up])

    f1_up = convBlock(up_conv_f2_up, 128)
    g2 = gating(f1_up, 64)
    a2 = attention_gate(g2, f1, 64)
    up_conv_f1_up = UpSampling2D(size=(2,2))(f1_up)
    concat_2 = Concatenate([a2, up_conv_f1_up])

    f0 = convBlock(up_conv_f1_up, 64)

    conv_final = Conv2D(1, 1)(f0)
    output = Activation('sigmoid')(conv_final)

    model = Model(inputs=[inputs], outputs=[output])
    return model




if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_attention_unet(input_shape)
    model.summary()
    # build_attention_unet(input_shape)
