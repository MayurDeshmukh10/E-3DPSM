import torch
import torch.nn as nn
EPS = 1.1920929e-07


class HeatMapJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatMapJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
                        
            if self.use_target_weight:
                tw = target_weight[:, idx]
                    
                loss_j= self.criterion(
                    heatmap_pred.mul(tw),
                    heatmap_gt.mul(tw)
                )
            else:
                loss_j= self.criterion(heatmap_pred, heatmap_gt)

            loss += loss_j
        
        if loss == 0:
            raise ValueError("No valid HeatMapJointsMSELoss")
        return loss.mean() / num_joints


class J3dMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(J3dMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):        
        num_joints = output.size(1)

        cnt = 0
        loss = 0
        for idx in range(num_joints):
            j3d_pred = output[:, idx]
            j3d_gt = target[:, idx]
            
            if self.use_target_weight:
                tw = target_weight[:, idx]
                
                loss += self.criterion(
                    j3d_pred.mul(tw),
                    j3d_gt.mul(tw)
                )
            else:
                loss += self.criterion(j3d_pred, j3d_gt)

        # import pdb; pdb.set_trace()
        # try:
        #     if loss == 0:
        #         print("No valid J3dMSELoss")
        # except:
        #     import pdb; pdb.set_trace()
        #     raise ValueError("No valid J3dMSELoss")
        return loss.mean() / num_joints

class BoneLengthLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(BoneLengthLoss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        num_joints = output.size(1)
        cnt = 0
        loss = 0
        B_up_1_pred = self.criterion(output[:, 5], output[:, 6]) + self.criterion(output[:, 2], output[:, 3])
        B_up_1_gt = self.criterion(target[:, 5], target[:, 6]) + self.criterion(target[:, 2], target[:, 3])

        B_up_2_pred = self.criterion(output[:, 6], output[:, 7]) + self.criterion(output[:, 3], output[:, 4])
        B_up_2_gt = self.criterion(target[:, 6], target[:, 7]) + self.criterion(target[:, 3], target[:, 4])

        B_low_1_pred = self.criterion(output[:, 12], output[:, 13]) + self.criterion(output[:, 8], output[:, 9])
        B_low_1_gt = self.criterion(target[:, 12], target[:, 13]) + self.criterion(target[:, 8], target[:, 9])

        B_low_2_pred = self.criterion(output[:, 13], output[:, 14]) + self.criterion(output[:, 9], output[:, 10])

        for idx in range(num_joints):
            j3d_pred = output[:, idx]
            j3d_gt = target[:, idx]
            
            if self.use_target_weight:
                tw = target_weight[:, idx]
                
                loss += self.criterion(
                    j3d_pred.mul(tw),
                    j3d_gt.mul(tw)
                )
            else:
                loss += self.criterion(j3d_pred, j3d_gt)

        return loss.mean() / num_joints

class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()

        # self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target, weight=None):
        if weight is None:
            return self.loss(output, target)
        else:
            # weight = weight.view(-1, 1, 1, 1)
            return self.loss(output * weight, target * weight)
  