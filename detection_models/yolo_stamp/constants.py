# shape of input image to YOLO
W, H =  448, 448
# grid size after last convolutional layer of YOLO
S = 7 
# anchors of YOLO model
ANCHORS = [[1.5340836003942058, 1.258424277571925],
 [1.4957766780406023, 2.2319885681948217],
 [1.2508985343739407, 0.8233350471152914]]
# number of anchors boxes
BOX = len(ANCHORS)
# maximum number of stamps on image
STAMP_NB_MAX = 10
# minimal confidence of presence a stamp in the grid cell
OUTPUT_THRESH = 0.7
# maximal iou score to consider boxes different
IOU_THRESH = 0.3
# path to folder containing images
IMAGE_FOLDER = './data/images'
# path to .cvs file containing annotations
ANNOTATIONS_PATH = './data/all_annotations.csv'
# standard deviation and mean of pixel values for normalization
STD = (0.229, 0.224, 0.225)
MEAN = (0.485, 0.456, 0.406)
# box color to show the bounding box on image
BOX_COLOR = (0, 0, 255)
