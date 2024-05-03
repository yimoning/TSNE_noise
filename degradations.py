import rawpy
import numpy as np
import glob, os
import imageio
import argparse
from PIL import Image as PILImage
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2

from scipy.io import loadmat
from scipy import ndimage
from scipy.signal import convolve2d
# import hdf5storage

import torch
import torch.nn as nn
import torch.nn.functional as F

from blur import apply_psf, add_blur
from noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from imutils import downsample_raw, convert_to_tensor

def extract_into_tensor(a, t, x_shape):
    # b, *_ = t.shape
    b = 1
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, alphas_cumprod, noise=None):

    if noise is None:
       noise = torch.randn_like(x_start)
       noise = noise * 0.001
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_alphas_cumprod_t = extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img)

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)
    values = [float(line.strip()) for line in open('noise.txt') if line.strip()]
    values = torch.tensor(values)
    t = torch.randint(1, 11, (1,))
    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise > 0.3:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)

    # if p_noise < 0.3:
    #     img = add_natural_noise(img)
    # elif p_noise < 0.8:
    #     img = add_heteroscedastic_gnoise(img)
    # else:
    #     img = q_sample(img, torch.tensor(t, dtype=int), values)
    img[img > 1] = 1  # Set values greater than 1 to 1
    img[img < 0] = 0

    return img