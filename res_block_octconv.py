import keras
import tensorflow as tf
from keras.layers import *
from . import *

def BN(data):
        x = BatchNormalization()(data)
        return x

def AC(data):
        x = Activation('relu')(data)
        return x

def BN_AC(data):
        x = BatchNormalization()(data)
        x = Activation('relu')(x)
        return x

def firstOctConv_BN_AC(data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out_hf = BN_AC(data=hf_data)
    out_lf = BN_AC(data=lf_data)
    return out_hf, out_lf

def lastOctConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out = BN_AC(data=conv)
    return out

def octConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out_hf = BN_AC(data=hf_data)
    out_lf = BN_AC(data=lf_data)
    return out_hf, out_lf


def firstOctConv_BN(data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out_hf = BN(data=hf_data)
    out_lf = BN(data=lf_data)
    return out_hf, out_lf

def lastOctConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out = BN(data=conv)
    return out

def octConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel_size, pad, strides=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel_size=kernel_size, pad=pad, strides=strides)
    out_hf = BN(data=hf_data)
    out_lf = BN(data=lf_data)
    return out_hf, out_lf




def Residual_Unit_norm(data, num_in, num_mid, num_out, name, first_block=False, strides=(1, 1), g=1):
    conv_m1 = Conv_BN_AC( data=data,    num_filter=num_mid, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m1' % name))
    conv_m2 = Conv_BN_AC( data=conv_m1, num_filter=num_mid, kernel_size=( 3,  3), pad=( 1,  1), name=('%s_conv-m2' % name), strides=strides, num_group=g)
    conv_m3 = Conv_BN( data=conv_m2, num_filter=num_out,   kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m3' % name))

    if first_block:
        data = Conv_BN( data=data, num_filter=num_out, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-w1' % name), strides=strides)

    outputs = merge.add([data, conv_m3])
    return AC(outputs)



def Residual_Unit_last(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, strides=(1, 1), g=1):
    hf_data_m, lf_data_m = octConv_BN_AC( hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m1' % name))
    conv_m2 = lastOctConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, name=('%s_conv-m2' % name), kernel_size=(3,3), pad=(1,1), strides=strides)
    conv_m3 = Conv_BN( data=conv_m2, num_filter=num_out,   kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m3' % name))

    if first_block:
        data = lastOctConv_BN(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, name=('%s_conv-w1' % name), kernel_size=(1,1), pad=(0,0), strides=strides)

    outputs = merge.add([data, conv_m3])
    outputs = AC(outputs)
    return outputs

def Residual_Unit_first(data, alpha, num_in, num_mid, num_out, name, first_block=False, strides=(1, 1), g=1):
    hf_data_m, lf_data_m = firstOctConv_BN_AC(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN_AC( hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, kernel_size=( 3,  3), pad=( 1,  1), name=('%s_conv-m2' % name), strides=strides, num_group=g)
    hf_data_m, lf_data_m = octConv_BN( hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid, num_filter_out=num_out,  kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m3' % name))

    if first_block:
        hf_data, lf_data = firstOctConv_BN( data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-w1' % name), strides=strides)

    hf_outputs = merge.add([hf_data, hf_data_m])
    lf_outputs = merge.add([lf_data, lf_data_m])

    hf_outputs = AC(hf_outputs)
    lf_outputs = AC(lf_outputs)
    return hf_outputs, lf_outputs

def Residual_Unit(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, strides=(1, 1), g=1):
    hf_data_m, lf_data_m = octConv_BN_AC( hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN_AC( hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, kernel_size=( 3,  3), pad=( 1,  1), name=('%s_conv-m2' % name), strides=strides, num_group=g)
    hf_data_m, lf_data_m = octConv_BN( hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid, num_filter_out=num_out,  kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-m3' % name))

    if first_block:
        hf_data, lf_data = octConv_BN( hf_data=hf_data, lf_data=lf_data,  alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, kernel_size=( 1,  1), pad=( 0,  0), name=('%s_conv-w1' % name), strides=strides)

    hf_outputs = merge.add([hf_data, hf_data_m])
    lf_outputs = merge.add([lf_data, lf_data_m])

    hf_outputs = AC(hf_outputs)
    lf_outputs = AC(lf_outputs)
    return hf_outputs, lf_outputs


