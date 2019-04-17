import keras
import tensorflow as tf
from keras.layers import *



def firstOctConv(data, settings, ch_in, ch_out, name, kernel_size=(1,1), pad=(0,0), strides=(1,1)):

    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out
    
    hf_data = data
    
    if strides == (2, 2):
        hf_data = AvgPool2D(kernel_size=(2,2), strides=(2,2), padding='valid')(hf_data)
    hf_conv = Conv2D(filters=hf_ch_out, kernel=kernel, strides=strides, padding=pad)(hf_data)
    hf_pool = AvgPool2D(kernel_size=(2,2), strides=(2,2), padding='valid')(hf_data)
    hf_pool_conv = Conv2D(filters=lf_ch_out, kernel_size=kernel_size, padding=pad, strides=strides)(hf_pool)

    out_h = hf_conv 
    out_l = hf_pool_conv 
    return out_h, out_l 





