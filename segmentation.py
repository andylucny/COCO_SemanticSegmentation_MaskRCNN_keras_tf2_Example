# https://keras.io/
# https://github.com/matterport/Mask_RCNN
# https://github-production-release-asset-2e65be.s3.amazonaws.com/107595270/872d3234-d21f-11e7-9a51-7b4bc8075835?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20200105%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200105T203020Z&X-Amz-Expires=300&X-Amz-Signature=9673a103dac995b8bb23ebc2d617ac25ac8d18d0413f76d9dad8a3659438a984&X-Amz-SignedHeaders=host&actor_id=12858649&response-content-disposition=attachment%3B%20filename%3Dmask_rcnn_coco.h5&response-content-type=application%2Foctet-stream
# https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
import numpy as np
import cv2

# load the class label names from disk, one label per line
classes = open("coco/coco_labels.txt").read().strip().split("\n")
count = len(classes)

# generate visually distinct colors for each class label
colors = [tuple(cv2.cvtColor(np.array([[[180*i//count,255,255]]],np.uint8),cv2.COLOR_HSV2BGR)[0,0]) for i in range(count)]
colors = [(int(a[0]),int(a[1]),int(a[2])) for a in colors]

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"
	# set the number of GPUs to use along with the number of images per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes 
	NUM_CLASSES = count

# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='coco')
model.load_weights("coco/mask_rcnn_coco.h5", by_name=True)

# load the input image, convert it from BGR to RGB channel and resize
image = cv2.imread("inputs/spz-more.png")
#width = 512
width = 1024
image = cv2.resize(image,(width,width*image.shape[0]//image.shape[1]))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform a forward pass of the network to obtain the results
out = model.detect([rgb], verbose=1)[0]

# loop over of the detected object's bounding boxes and masks
for i in range(0, out["rois"].shape[0]):
    # extract the class ID and mask for the current detection
    class_id = out["class_ids"][i]
    mask = out["masks"][:, :, i]
    # visualize the pixel-wise mask of the object
    color = colors[class_id]
    for c in range(3):
        image[:,:,c] = np.where(mask==1, image[:,:,c]//2 + color[c]//2, image[:,:,c])

# loop over the predicted scores and class labels
for i in range(0, len(out["scores"])):
	# extract the bounding box information, class ID, label, predicted probability, and visualization color
	startY, startX, endY, endX = out["rois"][i]
	class_id = out["class_ids"][i]
	label = classes[class_id]
	score = out["scores"][i]
	color = colors[class_id]
	# draw the bounding box, class label, and score of the object
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}: {:.3f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX,y), 0, 0.6, color, 2)

# show the output image
cv2.imwrite("outputs/result.png",image)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

