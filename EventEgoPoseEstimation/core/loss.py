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


class BoneLoss(nn.Module):
    heatmap_sequence = ["Head", # 0
                        "Neck", # 1
                        "Right_shoulder", # 2 
                        "Right_elbow", # 3
                        "Right_wrist", # 4
                        "Left_shoulder", # 5
                        "Left_elbow", # 6
                        "Left_wrist", # 7
                        "Right_hip", # 8
                        "Right_knee", # 9
                        "Right_ankle", # 10
                        "Right_foot", # 11
                        "Left_hip", # 12 
                        "Left_knee", # 13
                        "Left_ankle", #14
                        "Left_foot" # 15
                        ] 

                        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    kinematic_parents = [ 0, 0, 1, 2, 3, 1, 5, 6, 2, 8,  9, 10,  5, 12, 13, 14]

    print('Kinematic Parents:')
    for i in range(len(heatmap_sequence)):
        print(f'{heatmap_sequence[i]} -> {heatmap_sequence[kinematic_parents[i]]}')

    def __init__(self):
        super(BoneLoss, self).__init__()

        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, pose_predicted, pose_gt, ange_weight, length_weight):
        # New shapes:
        # pose_predicted: [T, B, 16, 3]
        # pose_gt:        [T, B, 16, 3]
        # ange_weight & length_weight: [T, B, 16, 1]
        
        T, B, J, D = pose_predicted.shape  # T=50, B=2, J=16, D=3

        # Merge time and batch dimensions to treat them as independent examples.
        pose_predicted = pose_predicted.view(T * B, J, D)  # [T*B, 16, 3]
        pose_gt = pose_gt.view(T * B, J, D)                # [T*B, 16, 3]
        
        # Similarly, merge and process weights.
        ange_weight = ange_weight.view(T * B, J, 1)        # [T*B, 16, 1]
        length_weight = length_weight.view(T * B, J, 1)      # [T*B, 16, 1]
        
        # Squeeze the last dimension to get [T*B, 16]
        ange_weight = ange_weight.squeeze(-1)   # [T*B, 16]
        length_weight = length_weight.squeeze(-1)  # [T*B, 16]
        
        # Average weights over joints to obtain a per-sample weight.
        ange_weight = torch.mean(ange_weight, dim=1)   # [T*B]
        length_weight = torch.mean(length_weight, dim=1)  # [T*B]
        
        # Compute bone vectors. Assumes self.kinematic_parents is defined such that:
        #   pose_predicted[:, self.kinematic_parents, :] gives each joint's parent's position.
        predicted_bone_vector = pose_predicted - pose_predicted[:, self.kinematic_parents, :]
        gt_bone_vector = pose_gt - pose_gt[:, self.kinematic_parents, :]
        
        # Exclude the root joint (first element).
        predicted_bone_vector = predicted_bone_vector[:, 1:, :]  # [T*B, num_bones, 3] where num_bones = J-1
        gt_bone_vector = gt_bone_vector[:, 1:, :]
        
        # Compute cosine similarity loss for bone directions.
        # Assuming self.cos_sim computes cosine similarity along the last dimension.
        cos_loss = 1 - self.cos_sim(predicted_bone_vector, gt_bone_vector)  # [T*B, num_bones]
        cos_loss = torch.sum(cos_loss, dim=1)   # Sum over bones -> [T*B]
        cos_loss = torch.mean(cos_loss * ange_weight)  # Weighted average over all examples.
        
        # Compute bone length loss.
        # predicted_bone_length = torch.norm(predicted_bone_vector, p=2, dim=-1)  # [T*B, num_bones]
        # gt_bone_length = torch.norm(gt_bone_vector, p=2, dim=-1)                # [T*B, num_bones]
        # bone_length_loss = torch.sum((predicted_bone_length - gt_bone_length) ** 2, dim=1)  # [T*B]
        # bone_length_loss = torch.mean(bone_length_loss * length_weight)
        
        return cos_loss