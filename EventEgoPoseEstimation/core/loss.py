import torch
import torch.nn as nn
EPS = 1.1920929e-07


class HeatMapJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatMapJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(1)
        num_joints = output.size(2)
        # heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        # heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        
        for idx in range(num_joints):
            heatmap_pred = output[:, :, idx, :, :]
            heatmap_gt = target[:, :, idx, :, :]
            # heatmap_pred = heatmaps_pred[idx].squeeze()
            # heatmap_gt = heatmaps_gt[idx].squeeze()
                        
            if self.use_target_weight:
                tw = target_weight[:, :, idx, :, None]

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


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        num_joints = output.size(2)
        loss = 0
        for idx in range(num_joints):
            joint_pred = output[:, :, idx]
            joint_gt = target[:, :, idx]
            
            if self.use_target_weight:
                tw = target_weight[:, :, idx]
                
                loss += self.criterion(
                    joint_pred.mul(tw),
                    joint_gt.mul(tw)
                )
            else:
                loss += self.criterion(joint_pred, joint_gt)

        return loss.mean() / num_joints

# class BoneLengthLoss(nn.Module):
#     def __init__(self, use_target_weight):
#         super(BoneLengthLoss, self).__init__()
#         self.criterion = nn.L1Loss(size_average=False)
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target):        
#         B_up_1_pred = torch.abs(output[:, :, 5, :] - output[:, :, 6, :]).sum(dim=-1) + torch.abs(output[:, :, 2, :] - output[:, :, 3, :]).sum(dim=-1)
#         B_up_1_gt = torch.abs(target[:, :, 5, :] - target[:, :, 6, :]).sum(dim=-1) + torch.abs(target[:, :, 2, :] - target[:, :, 3, :]).sum(dim=-1)

#         B_up_2_pred = torch.abs(output[:, :, 6, :] - output[:, :, 7, :]).sum(dim=-1) + torch.abs(output[:, :, 3, :] - output[:, :, 4, :]).sum(dim=-1)
#         B_up_2_gt = torch.abs(target[:, :, 6, :] - target[:, :, 7, :]).sum(dim=-1) + torch.abs(target[:, :, 3, :] - target[:, :, 4, :]).sum(dim=-1)

#         B_low_1_pred = torch.abs(output[:, :, 12, :] - output[:, :, 13, :]).sum(dim=-1) + torch.abs(output[:, :, 8, :] - output[:, :, 9, :]).sum(dim=-1)
#         B_low_1_gt = torch.abs(target[:, :, 12, :] - target[:, :, 13, :]).sum(dim=-1) + torch.abs(target[:, :, 8, :] - target[:, :, 9, :]).sum(dim=-1)

#         B_low_2_pred = torch.abs(output[:, :, 13, :] - output[:, :, 14, :]).sum(dim=-1) + torch.abs(output[:, :, 9, :] - output[:, :, 10, :]).sum(dim=-1)
#         B_low_2_gt = torch.abs(target[:, :, 13, :] - target[:, :, 14, :]).sum(dim=-1) + torch.abs(target[:, :, 9, :] - target[:, :, 10, :]).sum(dim=-1)

#         bone_length_up = torch.abs(B_up_1_pred - B_up_1_gt) + torch.abs(B_up_2_pred - B_up_2_gt)
#         bone_length_low = torch.abs(B_low_1_pred - B_low_1_gt) + torch.abs(B_low_2_pred - B_low_2_gt)

#         bone_length = torch.mean(bone_length_up + bone_length_low)

#         return bone_length

class BoneLengthLoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(BoneLengthLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.eps = 1e-6
        
        # Define bone connections as (parent, child) indices
        self.bone_pairs = [
            (2, 3),    # Right upper arm
            (3, 4),    # Right lower arm
            (5, 6),    # Left upper arm
            (6, 7),    # Left lower arm
            (8, 9),    # Right upper leg
            (9, 10),   # Right lower leg
            (12, 13),  # Left upper leg
            (13, 14),  # Left lower leg
        ]

    def forward(self, output, target, target_weight=None):
        total_loss = 0.0
        total_valid = 0.0

        for parent, child in self.bone_pairs:
            # Calculate bone lengths (L2 norm)

            pred_length = torch.norm(output[..., child, :] - output[..., parent, :], p=2, dim=-1)
            gt_length = torch.norm(target[..., child, :] - target[..., parent, :], p=2, dim=-1)
            
            # Calculate length difference (L1 loss)
            length_diff = torch.abs(pred_length - gt_length)
            
            # Calculate validity weights
            if self.use_target_weight and target_weight is not None:
                valid = target_weight[..., parent, :] * target_weight[..., child, :]
                valid = valid.squeeze()
            else:
                valid = torch.ones_like(length_diff)
            
            total_loss += (length_diff * valid).sum()
            total_valid += valid.sum()

        if total_valid < self.eps:
            return torch.tensor(0.0, device=output.device)
            
        return total_loss / (total_valid + self.eps)


class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target, weight=None):

        # output = torch.clamp(output, min=-88, max=88)

        if weight is None:
            return self.loss(output, target)
        else:
            # weight = weight.view(2, 2, 1, 1, 1)
            return self.loss(output * weight, target * weight)


class BoneOrientationLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(BoneOrientationLoss, self).__init__()
        # self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.eps = 1e-6

        self.bone_pairs = [
            (13, 14),  # Left lower leg
            (12, 13),  # Left upper leg
            (9, 10),   # Right lower leg
            (8, 9),    # Right upper leg
            (6, 7),    # Left lower arm
            (5, 6),    # Left upper arm
            (3, 4),    # Right lower arm
            (2, 3)     # Right upper arm
        ]


    def forward(self, output, target, target_weight=None):

        total_loss = 0.0
        total_valid = 0.0

        for parent, child in self.bone_pairs:
            # Calculate bone vectors (child - parent)
            pred_bone = output[..., child, :] - output[..., parent, :]
            gt_bone = target[..., child, :] - target[..., parent, :]

            # Compute cosine similarity
            dot_product = (pred_bone * gt_bone).sum(dim=-1)
            pred_norm = torch.norm(pred_bone, p=2, dim=-1)
            gt_norm = torch.norm(gt_bone, p=2, dim=-1)
            cosine_sim = dot_product / (pred_norm * gt_norm + self.eps)
            
            # Calculate bone loss (1 - cosine similarity)
            bone_loss = 1 - cosine_sim

            # Calculate bone validity weights
            if self.use_target_weight:
                valid = target_weight[..., parent] * target_weight[..., child]
            else:
                valid = torch.ones_like(bone_loss)
            
            total_loss += (bone_loss * valid).sum()
            total_valid += valid.sum()

        # Handle case with no valid bones
        if total_valid < self.eps:
            return torch.tensor(0.0, device=output.device)

        return total_loss / (total_valid + self.eps)
