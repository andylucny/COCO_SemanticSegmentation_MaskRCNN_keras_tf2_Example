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

https://github-production-release-asset-2e65be.s3.amazonaws.com/107595270/872d3234-d21f-11e7-9a51-7b4bc8075835?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20200105%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200105T203020Z&X-Amz-Expires=300&X-Amz-Signature=9673a103dac995b8bb23ebc2d617ac25ac8d18d0413f76d9dad8a3659438a984&X-Amz-SignedHeaders=host&actor_id=12858649&response-content-disposition=attachment%3B%20filename%3Dmask_rcnn_coco.h5&response-content-type=application%2Foctet-stream (download the model here)

https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
