# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['ToTensor', 'RandomPatch', 'AugMix', "MotionBlur", "Rotate1","JpegCompress"]

import math
import random
from collections import deque
from PIL import Image

import numpy as np
import torch

from .functional import to_tensor, augmentations
import albumentations as A

def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob           = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width  = mixture_width
        self.mixture_depth  = mixture_depth
        self.aug_severity   = aug_severity
        self.augmentations  = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.

        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)

class MotionBlur(object):
    """Blur the input image using a random-sized kernel.
    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, p=0.5):
        # super(Blur, self).__init__(always_apply, p)
        self.p = p
        self.blur_limit = blur_limit
        self.trans = A.MotionBlur(blur_limit=self.blur_limit,p=p)


    def __call__(self,image):
        t1 = np.array(image)
        t2 = self.trans(image=t1)["image"]
        return Image.fromarray(t2)

class Rotate1(object):
    """Blur the input image using a random-sized kernel.
    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, p=0.5):
        # super(Blur, self).__init__(always_apply, p)
        self.p = p
        self.trans = A.Rotate(p=p)


    def __call__(self,image):
        t1 = np.array(image)
        t2 = self.trans(image=t1)["image"]
        return Image.fromarray(t2)

class JpegCompress(object):
    """Decrease Jpeg compression of an image.
    Args:
        quality_lower (float): lower bound on the jpeg quality. Should be in [0, 100] range
        quality_upper (float): upper bound on the jpeg quality. Should be in [0, 100] range
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, quality_lower=10, quality_upper=50, always_apply=False, p=0.5):
        # super(Blur, self).__init__(always_apply, p)
        self.p = p
        self.trans = A.JpegCompression(quality_lower, quality_upper,p=p)


    def __call__(self,image):
        t1 = np.array(image)
        t2 = self.trans(image=t1)["image"]
        return Image.fromarray(t2)
