from torch import nn

import torch
import torch.nn.functional as F
from bpgm.model.models import Vgg19
from torchvision.models import vgg16
from piqa import SSIM


    
LABEL_REAL, LABEL_FAKE = 1.0, 0.0


class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, y_hat, label):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")
        
        y_true = torch.full(y_hat.size(), label, device="cuda:0")
        return self.loss(y_hat, y_true)
    
def GMMLoss(parse_cloth, warp_coarse_cloth, warp_fine_cloth):
    
    loss_l1 = nn.L1Loss()
    loss_vgg = VGGLoss()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    lambda4 = 0.5
    lambda5 = 0.5
    k = 3
    
    ls0 = loss_l1(parse_cloth, warp_coarse_cloth)
    ls1 = loss_l1(parse_cloth, warp_fine_cloth)
    
    lpush = k*ls1 - loss_l1(warp_fine_cloth, warp_coarse_cloth)
    
    v0 = loss_vgg(warp_coarse_cloth, parse_cloth)
    v1 = loss_vgg(warp_fine_cloth, parse_cloth)
    cos = torch.mean(cos_sim(v0, v1))
    lalign = torch.pow(cos-1, 2)
    
    lpgm = lambda4*lpush + lambda5*lalign
    loss = lambda1*ls0 + lambda2*ls1 + lambda3*lpgm
    
    return loss



def ssim_loss(x, y, window_size=11, sigma=1.5, size_average=True):
    if len(x.shape) != 4:
        raise ValueError('Input tensors should have shape (batch_size, channels, height, width)')
    if not x.shape == y.shape:
        raise ValueError('Input tensors should have the same shape')
    if not window_size % 2 == 1:
        raise ValueError('Window size should be an odd number')
    # Pad the input tensors to handle border pixels
    padding = window_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    y = F.pad(y, (padding, padding, padding, padding), mode='reflect')
    # Compute mean and variance of input tensors
    mu_x = F.avg_pool2d(x, window_size, stride=1)
    mu_y = F.avg_pool2d(y, window_size, stride=1)
    sigma_x = F.avg_pool2d(x ** 2, window_size, stride=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, window_size, stride=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1) - mu_x * mu_y
    # Compute SSIM index
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    # Compute MS-SSIM index
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda()
    msssim = torch.prod((ssim ** weights).view(-1, 5), dim=1)
    # Compute loss
    loss = 1 - msssim
    if size_average:
        loss = torch.mean(loss)
    return loss
        #     # criterionNew = MyNewLoss()

        # # Calculate the total loss
        # loss_cloth = criterionL1(warped_cloth, im_c) + 0.1 * criterionVGG(warped_cloth, im_c) + 0.5 * criterionNew(warped_cloth, im_c)

        # # Calculate the total loss
        # loss_mask = criterionL1(warped_mask, im_cm) * 0.1

        # # Add up the losses
        # loss = loss_cloth + loss_mask
        # USE LIEK THIS







class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)