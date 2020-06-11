import torch
from torch import nn

class GazeLoss(nn.Module):
    def __init__(self):
        super(GazeLoss, self).__init__()

    def forward(self, gaze_gt, hp_gt, gaze_pred, hp_pred):
        # weight_hp = torch.sum(1 - torch.cos(hp_gt - hp_pred), dim=1)
        # cos_distant = torch.sum(1 - torch.cos(gaze_gt - gaze_pred), dim=1)
        L2_hp_weight = torch.sum((hp_gt - hp_pred) * (hp_gt - hp_pred) , dim=1)
        # L1_gaze = torch.sum(torch.abs(gaze_gt - gaze_pred) , dim=1)
        L2_gaze = torch.sum((gaze_gt - gaze_pred) * (gaze_gt - gaze_pred) , dim=1)
        # return torch.mean(0.25* L2_hp_weight + (L2_gaze))
        return torch.mean(0.2*L2_hp_weight +  (L2_gaze))
        # return torch.mean((L2_gaze))


class L2Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, g_gt, g, train_batchsize):
        return torch.mean(torch.sum((g_gt - g) * (g_gt - g), dim=1))

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, g_gt, g, train_batchsize):
        return torch.mean(torch.abs(g_gt - g) , dim=1)

class L1andL2Loss(nn.Module):
    def __init__(self):
        super(L1andL2Loss, self).__init__()

    def forward(self, g_pred, g_gt):
        L1 = torch.mean(torch.sum(torch.abs(g_gt - g_pred) , dim=1))
        L2 = torch.mean(torch.sum((g_gt - g_pred) * (g_gt - g_pred) , dim=1))
        return L1 + L2
       
class LogCosh(nn.Module):
    ''' log cosh loss'''
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, gaze_gt, hp_gt, gaze_pred, hp_pred):
        # weight_hp = torch.sum(1 - torch.cos(hp_gt - hp_pred), dim=1)
        # cos_distant = torch.sum(1 - torch.cos(gaze_gt - gaze_pred), dim=1)
        hp_losls = torch.sum(torch.log(torch.cosh(hp_gt - hp_pred)) , dim=1)
        # L1_gaze = torch.sum(torch.abs(gaze_gt - gaze_pred) , dim=1)
        gaze_loss = torch.sum(torch.log(torch.cosh(gaze_gt - gaze_pred)) , dim=1)
        # return torch.mean(0.25* L2_hp_weight + (L2_gaze))
        return torch.mean(0.2*hp_losls +  (gaze_loss))
