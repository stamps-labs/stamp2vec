import torch
import torch.nn as nn
from utils import *

"""
    Class for loss for training YOLO model.

    Argmunets:
    h_coord: weight for loss related to coordinates and shapes of box
    h__noobj: weight for loss of predicting presence of box when it is absent.
"""
class YOLOLoss(nn.Module):
    def __init__(self, h_coord=0.5, h_noobj=2., h_shape=2., h_obj=10.):
        super().__init__()
        self.h_coord = h_coord
        self.h_noobj = h_noobj
        self.h_shape = h_shape
        self.h_obj = h_obj
    
    def square_error(self, output, target):
        return (output - target) ** 2

    def forward(self, output, target):
        
        pred_xy, pred_wh, pred_obj = yolo_head(output)
        gt_xy, gt_wh, gt_obj = process_target(target)

        pred_ul = pred_xy - 0.5 * pred_wh
        pred_br = pred_xy + 0.5 * pred_wh
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]

        gt_ul = gt_xy - 0.5 * gt_wh
        gt_br = gt_xy + 0.5 * gt_wh
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]

        intersect_ul = torch.max(pred_ul, gt_ul)
        intersect_br = torch.min(pred_br, gt_br)
        intersect_wh = intersect_br - intersect_ul
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        iou = intersect_area / (pred_area + gt_area - intersect_area)
        max_iou = torch.max(iou, dim=3, keepdim=True)[0]
        best_box_index = torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
        gt_box_conf = best_box_index * gt_obj

        xy_loss = (self.square_error(pred_xy, gt_xy) * gt_box_conf).sum()
        wh_loss = (self.square_error(pred_wh, gt_wh) * gt_box_conf).sum()
        obj_loss = (self.square_error(pred_obj, gt_obj) * gt_box_conf).sum()
        noobj_loss = (self.square_error(pred_obj, gt_obj) * (1 - gt_box_conf)).sum()

        total_loss = self.h_coord * xy_loss + self.h_shape * wh_loss + self.h_obj * obj_loss + self.h_noobj * noobj_loss
        return total_loss