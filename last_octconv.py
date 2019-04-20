import keras
import tensorflow as tf
from keras.layers import *



def lastOctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1,1), pad=(0,0), stride=(1,1)):

    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))
 
    if stride == (2, 2):
        hf_data = AvgPool2D(kernel_size=(2,2), strides=(2,2), padding='valid')(hf_data)
        lf_data = AvgPool2D(kernel_size=(2,2), strides=(2,2), padding='valid')(lf_data)

    hf_conv = Conv2D(filters=hf_ch_out, kernel_size=kernel_size, padding=pad, strides=strides)(hf_data)
    lf_conv = Conv2D(filters=hf_ch_out, kernel_size=kernel_size, padding=pad, strides=strides)(lf_data)

    out_h = hf_conv + lf_conv

    return out_h 

