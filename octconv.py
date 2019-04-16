import keras
import tensorflow as tf
from keras.layers import *
from functools import partial


def OctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1,1), pad=(0,0), stride=(1,1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    if stride == (2, 2):
        hf_data = AvgPool2D(kernel=(2,2), stride=(2,2), padding='valid')(hf_data)

    hf_conv = Conv2D( filters=hf_ch_out, kernel=kernel, padding=pad, stride=(1,1))(hf_data)
    hf_pool = AvgPool2D(kernel=(2,2), stride=(2,2), padding='valid')(hf_data)
    hf_pool_conv = Conv2D(filters=lf_ch_out, kernel=kernel, padding=pad, stride=(1,1))(hf_pool)
    lf_conv = Conv2D(filters=hf_ch_out, kernel=kernel, , padding=pad, stride=(1,1))(lf_data)

    if stride == (2, 2):
        lf_upsample = lf_conv
        lf_down = AvgPool2D(kernel=(2,2), stride=(2,2), padding='valid')(lf_data)
    else:
        lf_upsample = UpSampling2D(size=(2, 2), interpolation='nearest')(lf_conv)
        lf_down = lf_data

    lf_down_conv = Conv2D(filters=lf_ch_out, kernel=kernel, stride=(1,1), padding=pad)(lf_down)

    out_h = hf_conv + lf_upsample
    out_l = hf_pool_conv + lf_down_conv

    return out_h, out_l 