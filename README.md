hardcaffe
============================================
## dependencies update
* support for protoc version 3.0 or higher
* supoort for cudnn version 6.0 or higher

## layer update
* support for ResizeBilinear layer(equal to tf.image.resize_bilinear operator)
* support for Relu6 layer(equal to tf.relu6 operator)
* support for shuffle layer
* support for depthwise convolution

## SSD detection
* support for soft-nms
* support for focal loss


## TODO
1. modidy caffe/cudnn symmetric padding to tensorflow asymmetric padding -> use pad + slice




## Contributor
RuiminChen https://github.com/RuiminChen/Caffe-MobileNetV2-ReLU6

