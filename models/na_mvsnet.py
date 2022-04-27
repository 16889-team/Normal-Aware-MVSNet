import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from torch.autograd import Variable


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)[:, :, :, :, :-1]
        x = conv2 + self.conv9(x)[:, :, :, :, :-1]
        x = conv0 + self.conv11(x)[:, :, :, :, :-1]
        x = self.prob(x)
        return x


class NormalNet(nn.Module):
    def __init__(self):
        super(NormalNet, self).__init__()
        self.pool1 = ConvBnReLU3D(in_channels=32, out_channels=32, kernel_size=(2, 3, 3), stride=(2, 1, 1),
                                  pad=(0, 1, 1))
        self.pool2 = ConvBnReLU3D(in_channels=32, out_channels=32, kernel_size=(2, 3, 3), stride=(2, 1, 1),
                                  pad=(0, 1, 1))
        self.pool3 = ConvBnReLU3D(in_channels=32, out_channels=32, kernel_size=(2, 3, 3), stride=(2, 1, 1),
                                  pad=(0, 1, 1))

        self.wc0 = nn.Sequential(
            ConvBnReLU3D(32 + 3, 32, 3, 1, 1),
            ConvBnReLU3D(32, 32, 3, 1, 1),
        )

        self.cconvs_nfea = nn.Sequential(
            ConvText(64, 32, 3, 1, 1), ConvText(32, 32, 3, 1, 1)
        )
        self.n_convs0 = nn.Sequential(
            ConvText(32, 96, 3, 1, 1),
            ConvText(96, 96, 3, 1, 2),
            ConvText(96, 96, 3, 1, 4),
            ConvText(96, 64, 3, 1, 8),
            ConvText(64, 64, 3, 1, 16)
        )
        self.n_convs1 = nn.Sequential(
            ConvText(64, 32, 3, 1, 1), ConvText(32, 3, 3, 1, 1)
        )

    def forward(self, feature, no_pool=False):
        if no_pool:
            feature = self.pool1(self.wc0(feature))
        else:
            feature = self.pool3(self.pool2(self.pool1(self.wc0(feature))))
        slices = []

        B, _, D, H, W = feature.size()
        nmap = torch.zeros((B, 3, H, W)).type_as(feature)
        for i in range(D):
            normal_feature = self.n_convs0(feature[:, :, i])
            slices.append(self.n_convs1(normal_feature))
            if i == 0:
                nfea_conf = self.cconvs_nfea(normal_feature).clone()
            else:
                nfea_conf = nfea_conf + self.cconvs_nfea(normal_feature)
            nmap += slices[-1]
        nmap_nor = F.normalize(nmap, dim=1)
        nmap_nor = nmap_nor.detach()
        return nmap_nor


class NA_MVSNet(nn.Module):
    def __init__(self, min_depth=1.4):
        super(NA_MVSNet, self).__init__()
        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        self.normal_regression = NormalNet()
        self.min_depth = min_depth

    def forward(self, imgs, proj_matrices, intrinsics_inv, depth_values):
        ndepths = depth_values.shape[1]

        imgs = imgs.permute(0, 1, 4, 2, 3)
        imgs = torch.unbind(imgs, 1)  # list len = nviews  [B, height, width, 3]
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. normal regression
        # concat cost volume with world coordinates
        b, ch, d, h, w = volume_variance.size()
        with torch.no_grad():
            intrinsics_inv[:, :2, :2] = intrinsics_inv[:, :2, :2] * (4)
            disp2depth = Variable(torch.ones(b, h, w).cuda() * self.min_depth * ndepths).cuda()
            disps = Variable(
                torch.linspace(0, ndepths - 1, ndepths).view(1, ndepths, 1, 1).expand(b, ndepths, h, w)).type_as(
                disp2depth)
            depth = disp2depth.unsqueeze(1) / (disps + 1e-16)
            world_coord = pixel2cam(depth, intrinsics_inv)
            world_coord = world_coord.squeeze(-1)  # B x 3 x D x H x W
            world_coord = world_coord / (2 * num_depth * self.min_depth)

        world_coord = world_coord.clamp(-1, 1)
        world_coord = torch.cat((world_coord.clone(), volume_variance), dim=1)  # B x (3+F) x D x H x W
        world_coord = world_coord.contiguous()
        nmap = self.normal_regression(world_coord)

        # upsampel to original size
        nmap = F.interpolate(nmap, [img_height, img_width], mode='bilinear', align_corners=False)
        nmap = nmap.permute(0, 2, 3, 1)
        nmap = F.normalize(nmap, dim=-1)

        # step 4. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        cost_reg = F.upsample(cost_reg, [num_depth, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume,
                                           depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                     dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. depth map refinement
        return {"depth": depth, "normal": nmap, "photometric_confidence": photometric_confidence}


class ConsLoss(nn.Module):
    def __init__(self):
        super(ConsLoss, self).__init__()
        self.sobel_kernel = None

    def get_grad_1(self, depth):
        edge_kernel_x = torch.from_numpy(
            np.array([[1 / 8, 0, -1 / 8], [1 / 4, 0, -1 / 4], [1 / 8, 0, -1 / 8]])).type_as(depth)
        edge_kernel_y = torch.from_numpy(
            np.array([[1 / 8, 1 / 4, 1 / 8], [0, 0, 0], [-1 / 8, -1 / 4, -1 / 8]])).type_as(depth)
        self.sobel_kernel = torch.cat((edge_kernel_x.view(1, 1, 3, 3), edge_kernel_y.view(1, 1, 3, 3)), dim=0)
        self.sobel_kernel.requires_grad = False
        grad_depth = torch.nn.functional.conv2d(depth, self.sobel_kernel, padding=1)

        return -1 * grad_depth

    def get_grad_2(self, depth, nmap, intrinsics_var):
        p_b, _, p_h, p_w = depth.size()
        c_x = p_w / 2
        c_y = p_h / 2
        p_y = torch.arange(0, p_h).view(1, p_h, 1).expand(p_b, p_h, p_w).type_as(depth) - c_y
        p_x = torch.arange(0, p_w).view(1, 1, p_w).expand(p_b, p_h, p_w).type_as(depth) - c_x

        nmap_z = nmap[:, 2, :, :]
        nmap_z_mask = (nmap_z == 0)
        nmap_z[nmap_z_mask] = 1e-10
        nmap[:, 2, :, :] = nmap_z
        n_grad = nmap[:, :2, :, :].clone()
        n_grad = n_grad / (nmap[:, 2, :, :].unsqueeze(1))

        grad_depth = -n_grad.clone() * depth.clone()

        fx = intrinsics_var[:, 0, 0].clone().view(-1, 1, 1)
        fy = intrinsics_var[:, 1, 1].clone().view(-1, 1, 1)
        f = torch.cat((fx.unsqueeze(1), fy.unsqueeze(1)), dim=1)

        grad_depth = grad_depth / f

        denom = (1 + p_x * (n_grad[:, 0, :, :]) / fx + p_y * (n_grad[:, 1, :, :]) / fy)
        denom[denom == 0] = 1e-10
        grad_depth = grad_depth / denom.view(p_b, 1, p_h, p_w)

        return grad_depth

    def forward(self, depth, gt_depth, nmap, intrinsics_var, mask):
        depth = depth.unsqueeze(1)
        gt_depth = gt_depth.unsqueeze(1)

        nmap = nmap.permute(0, 3, 1, 2)

        true_grad_depth_1 = self.get_grad_1(gt_depth) * 100
        grad_depth_1 = self.get_grad_1(depth) * 100
        grad_depth_2 = self.get_grad_2(depth, nmap.clone(), intrinsics_var) * 100

        mask = mask.unsqueeze(1)

        mask = (abs(true_grad_depth_1) < 1).type_as(mask) & (mask)
        mask = (abs(grad_depth_1) < 5).type_as(mask) & (abs(grad_depth_2) < 5).type_as(mask) & (mask)
        mask.detach_()

        return F.smooth_l1_loss(grad_depth_1[mask], grad_depth_2[mask])


def na_mvsnet_loss(items, pred_depth, pred_normal, args):
    """
    :param items: dataloader item
    :param pred_depth: [B, height, width]
    :param pred_normal: [B, height, width, 3]
    :return: LOSS
    """
    depth_ref = items['depth_ref']
    depth_src = items['depth_src']
    normal_ref = items['normal_ref']
    normal_src = items['normal_src']
    mask_ref = items['mask_ref']
    mask_src = items['mask_src']
    grid = items['grid']
    e = items['extrinsics']
    i_inv = items['intrinsics_inv']  # [B, 3, 3]

    pred_normal_src, normal_masks = remap_normal(pred_normal, grid, e[:, 0], e[:, 1:])  # [B, N, height, width, 3]
    pred_depth_src, depth_masks = remap_depth(pred_depth, grid)

    mask_ref = mask_ref == True
    mask_src = mask_src == True

    mask_src_depth = mask_src * depth_masks
    mask_src_normal = mask_src * normal_masks

    loss_depth_ref = F.smooth_l1_loss(pred_depth[mask_ref], depth_ref[mask_ref], reduction='mean')
    loss_depth_src = F.smooth_l1_loss(pred_depth_src[mask_src_depth], depth_src[mask_src_depth], reduction='mean')
    loss_normal_ref = F.smooth_l1_loss(pred_normal[mask_ref], normal_ref[mask_ref], reduction='mean')
    loss_normal_src = F.smooth_l1_loss(pred_normal_src[mask_src_normal], normal_src[mask_src_normal], reduction='mean')

    # fix nan loss issue
    if torch.isnan(loss_depth_src):
      loss_depth_src = 0
    if torch.isnan(loss_normal_src):
      loss_normal_src = 0

    cons_loss = ConsLoss().cuda()
    cons_loss_ref = cons_loss(pred_depth, depth_ref, pred_normal, i_inv, mask_ref)

    loss = loss_depth_ref + args.beta * loss_normal_ref + args.alpha * (
            loss_depth_src + args.beta * loss_normal_src) + args.gamma * cons_loss_ref

    return loss
