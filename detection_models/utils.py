import torch
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .constants import *


def output_tensor_to_boxes(boxes_tensor):
    """
        Converts the YOLO output tensor to list of boxes with probabilites.

        Arguments:
        boxes_tensor -- tensor of shape (S, S, BOX, 5)

        Returns:
        boxes -- list of shape (None, 5)

        Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
        For example, the actual output size of scores would be (10, 5) if there are 10 boxes
    """
    cell_w, cell_h = W/S, H/S
    boxes = []
    
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                anchor_wh = torch.tensor(ANCHORS[b])
                data = boxes_tensor[i,j,b]
                xy = torch.sigmoid(data[:2])
                wh = torch.exp(data[2:4])*anchor_wh
                obj_prob = torch.sigmoid(data[4])
                
                if obj_prob > OUTPUT_THRESH:
                    x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h, obj_prob]
                    boxes.append(box)
    return boxes


def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.show()


def plot_normalized_img(img, std=STD, mean=MEAN, size=(7,7)):
    mean = mean if isinstance(mean, np.ndarray) else np.array(mean)
    std = std if isinstance(std, np.ndarray) else np.array(std)
    plt.figure(figsize=size)
    plt.imshow((255. * (img * std + mean)).astype(np.uint))
    plt.show()
    

def visualize_bbox(image: Image, prediction):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(prediction):
        x1, y1, x2, y2 = box.cpu()
        draw = ImageDraw.Draw(img)
        text_w, text_h = draw.textsize(str(i + 1))
        label_y = y1 if y1 <= text_h else y1 - text_h
        draw.rectangle((x1, y1, x2, y2), outline='red')
        draw.rectangle((x1, label_y, x1+text_w, label_y+text_h), outline='red', fill='red')
        draw.text((x1, label_y), str(i + 1), fill='white')
    return img


def read_data(annotations=Path(ANNOTATIONS_PATH)):
    """
        Reads annotations data from .csv file. Must contain columns: image_name, bbox_x, bbox_y, bbox_width, bbox_height.

        Arguments:
        annotations_path -- string or Path specifying path of annotations file

        Returns:
        data -- list of dictionaries containing path, number of boxes and boxes itself
    """
    data = []

    boxes = pd.read_csv(annotations)
    image_names = boxes['image_name'].unique()

    for image_name in image_names:
        cur_boxes = boxes[boxes['image_name'] == image_name]
        img_data = {
            'file_path': image_name,
            'box_nb': len(cur_boxes),
            'boxes': []}
        stamp_nb = img_data['box_nb']
        if stamp_nb <= STAMP_NB_MAX:
            img_data['boxes'] = cur_boxes[['bbox_x', 'bbox_y','bbox_width','bbox_height']].values
        data.append(img_data)
    return data

def xywh2xyxy(x):
    """
        Converts xywh format to xyxy

        Arguments:
        x -- torch.Tensor or np.array (xywh format)

        Returns:
        y -- torch.Tensor or np.array (xyxy)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2] 
    y[..., 3] = x[..., 1] + x[..., 3] 
    return y

def boxes_to_tensor(boxes):
    """
        Convert list of boxes (and labels) to tensor format
        
        Arguments:
        boxes -- list of boxes

        Returns:
        boxes_tensor -- tensor of shape (S, S, BOX, 5)
    """
    boxes_tensor = torch.zeros((S, S, BOX, 5))
    cell_w, cell_h = W/S, H/S
    for i, box in enumerate(boxes):
        x, y, w, h = box
        # normalize xywh with cell_size
        x, y, w, h = x / cell_w, y / cell_h, w / cell_w, h / cell_h
        center_x, center_y = x + w / 2, y + h / 2
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        if grid_x < S and grid_y < S:
            boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(BOX * [[center_x - grid_x, center_y - grid_y, w, h]])
            boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(BOX * [1.])
    return boxes_tensor


def target_tensor_to_boxes(boxes_tensor, output_threshold=OUTPUT_THRESH):
    """
        Recover target tensor (tensor output of dataset) to bboxes.
        Arguments:
            boxes_tensor -- tensor of shape (S, S, BOX, 5)
        Returns:
            boxes -- list of boxes, each box is [x, y, w, h]
    """
    cell_w, cell_h = W/S, H/S
    boxes = []
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                data = boxes_tensor[i,j,b]
                x_center,y_center, w, h, obj_prob = data[0], data[1], data[2], data[3], data[4]
                if obj_prob > output_threshold:
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h]
                    boxes.append(box)
    return boxes    


def overlap(interval_1, interval_2):
    """
        Calculates length of overlap between two intervals.

        Arguments:
        interval_1 -- list or tuple of shape (2,) containing endpoints of the first interval
        interval_2 -- list or tuple of shape (2, 2) containing endpoints of the second interval

        Returns:
        overlap -- length of overlap
    """
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3


def compute_iou(box1, box2):
    """
        Compute IOU between box1 and box2.

        Argmunets:
        box1 -- list of shape (5, ). Represents the first box
        box2 -- list of shape (5, ). Represents the second box
        Each box is [x, y, w, h, prob]

        Returns:
        iou -- intersection over union score between two boxes
    """
    x1,y1,w1,h1 = box1[0], box1[1], box1[2], box1[3]
    x2,y2,w2,h2 = box2[0], box2[1], box2[2], box2[3]

    area1, area2 = w1*h1, w2*h2
    intersect_w = overlap((x1,x1+w1), (x2,x2+w2))
    intersect_h = overlap((y1,y1+h1), (y2,y2+w2))
    if intersect_w == w1 and intersect_h == h1 or intersect_w == w2 and intersect_h == h2:
        return 1.
    intersect_area = intersect_w*intersect_h
    iou = intersect_area/(area1 + area2 - intersect_area)
    return iou


def nonmax_suppression(boxes, iou_thresh = IOU_THRESH):
    """
        Removes ovelap bboxes

        Arguments:
        boxes -- list of shape (None, 5)
        iou_thresh -- maximal value of iou when boxes are considered different
        Each box is [x, y, w, h, prob]

        Returns:
        boxes -- list of shape (None, 5) with removed overlapping boxes
    """
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > iou_thresh:
                boxes[j][4] = 0
    boxes = [box for box in boxes if box[4] > 0]
    return boxes


    
def yolo_head(yolo_output):
    """
        Converts a yolo output tensor to separate tensors of coordinates, shapes and probabilities.

        Arguments:
        yolo_output -- tensor of shape (batch_size, S, S, BOX, 5)

        Returns:
        xy -- tensor of shape (batch_size, S, S, BOX, 2) containing coordinates of centers of found boxes for each anchor in each grid cell
        wh -- tensor of shape (batch_size, S, S, BOX, 2) containing width and height of found boxes for each anchor in each grid cell
        prob -- tensor of shape (batch_size, S, S, BOX, 1) containing the probability of presence of boxes for each anchor in each grid cell
    """
    xy = torch.sigmoid(yolo_output[..., 0:2])
    anchors_wh = torch.tensor(ANCHORS, device=yolo_output.device).view(1, 1, 1, len(ANCHORS), 2)
    wh = torch.exp(yolo_output[..., 2:4]) * anchors_wh
    prob = torch.sigmoid(yolo_output[..., 4:5])
    return xy, wh, prob

def process_target(target):
    xy = target[..., 0:2]
    wh = target[..., 2:4]
    prob = target[..., 4:5]
    return xy, wh, prob