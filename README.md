# COCO_SemanticSegmentation_MaskRCNN_keras_tf2_Example

Mask-RCNN COCO model adjusted for Keras and Tensorflow 2.x

Example of semantic segmentation

Having CUDA 9.0 operational with CuDNN 7.6, install Python 3.6 or 3.7

download the model from link below and store it into the coco directory

save your picture as spz-more.png to inputs

pip install virtualenv

mkvirtualenv keras

pip install -r requirement.txt

workon keras

python segmentation.py

Based on:

https://keras.io/

https://github.com/matterport/Mask_RCNN

https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 (download the model here)

https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
