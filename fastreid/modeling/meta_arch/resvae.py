from torch import nn
import torch
import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.modeling.backbones.build import build_backbone
from fastreid.modeling.heads.build import build_heads
from fastreid.modeling.losses import *
from fastreid.modeling.losses.variance_loss import VarianceLoss
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

class Uncertainty(nn.Module):
    def __init__(self,inplanes):
        super().__init__()
        self.sample_number = 5
        self.get_sig = nn.Conv2d(in_channels=inplanes,out_channels=inplanes,kernel_size=1,stride=1,padding=0,bias=True)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        sig = F.softplus(self.get_sig(x)+1e-10)
        mu = x
        sample_dist = torch.distributions.Normal(loc=0,scale=1)
        B,C,H,W = sig.size()
        sample_features = []
        if self.training:
            for _ in range(self.sample_number):
                sample_feature = mu + sample_dist.sample() * sig
                sample_features.append(sample_feature)
        sig_z = sig.view(B,C,-1)
        attention = 1 - self.softmax(sig_z)
        mu_z = mu.view(B,C,-1)
        mus = torch.mul(mu_z,attention)
        mus = mus.view(B,C,H,W)
        return sample_features,mu,sig,attention.view(B,C,H,W)

@META_ARCH_REGISTRY.register()
class ResVAE(nn.Module):
    """
    cao ni ma de vae
    """
    @configurable
    def __init__(self,
            *,
            backbone,
            heads,
            oim_loss,
            pixel_mean,
            pixel_std,
            loss_kwargs=None):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.loss_kwargs = loss_kwargs
        self.oim_loss = oim_loss
        self.varloss = VarianceLoss()
        self.uncertainty = Uncertainty(2048)
        self.register_buffer('pixel_mean',torch.Tensor(pixel_mean).view(1,-1,1,1),False)
        self.register_buffer('pixel_std',torch.Tensor(pixel_std).view(1,-1,1,1),False)

    @classmethod
    def from_config(cls, cfg):
        oim_loss = OIMLoss(cfg.MODEL.BACKBONE.FEAT_DIM, cfg.MODEL.HEADS.NUM_CLASSES) #tmp moment
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'oim_loss': oim_loss,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },
                    'oim': {
                        'scale': cfg.MODEL.LOSSES.OIM.SCALE
                    }
                }
        }
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            samples,mu,sig,attention = self.uncertainty(features)
            l = len(samples)
            samples = torch.cat(samples,dim=0)

            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            logits = self.heads(samples)["cls_outputs"]

            loss_var = self.varloss(mu,sig,[logits],targets.repeat(5))

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            losses.update(loss_var=loss_var)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if "OimLoss" in loss_names:
            oim_kwargs = self.loss_kwargs.get('oim')
            loss_dict['loss_oim'] = self.oim_loss(
                pred_features,
                gt_labels) * oim_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict

