![](https://img.shields.io/badge/language-python_keras-orange.svg)
![](https://img.shields.io/badge/progress-70-green.svg)
[![](https://img.shields.io/badge/reference-paper-blue.svg)](https://arxiv.org/abs/1904.05049)
# OCTAVE_CONVOLUTION

This is a keras implementation of the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.](https://arxiv.org/abs/1904.05049)


### Separating the low and high spatial frequency signal


![](fig/fig1.png)
![](fig/fig2.png)

#### The spatial frequency model shows that natural image can be decomposed into a low and a high spatial frequency part. The output maps of a convolution layer can also be factorized and grouped by their spatial frequency. The proposed multi-frequency feature representation stores the smoothly changing, low-frequency maps in a low-resolution tensor to reduce spatial redundancy. The proposed Octave Convolution operates directly on this representation. It updates the information for each group and further enables information exchange between groups.

![](fig/octave.png)
