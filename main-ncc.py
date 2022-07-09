# -*- coding: utf-8 -*-
"""
Particle Image Velocimetry Using pyTorch implemented Normalized Cross Correlation
by: @erfanhamdi Jan 2022
"""
import yaml

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import utils

# load the config from yaml file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Image transformations using pytorch
image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomVerticalFlip(p=1),
    # transforms.Pad(padding=81, padding_mode='edge'),
    transforms.ToTensor(),
])

# Input images
img_1 = Image.open(config["img_1_address"])
img_1 = image_transforms(img_1)

img_2 = Image.open(config["img_2_address"])
img_2 = image_transforms(img_2)

# Algorithm Params
# Spatial Scale [m/pixel]
l_scale = config["l_scale"]
# Temporal Scale 1/frame_rate [s/frame]
t_scale = config["t_scale"]
# Interrodation Windows Sizes (pixel)
iw_size = config["iw_size"]
# Search Windows Sizes (sw > iw) (pixel)
sw_size = config["sw_size"]
# Interrogation window overlap percentage
overlap = config["overlap"]

batch, height, width = img_1.shape

# Changing the shape of windows from even to odd
iw_size = int( 2 * np.floor( (iw_size + 1) / 2 ) - 1 )
sw_size = int( 2 * np.floor( (sw_size + 1) / 2 ) - 1 )
# Calculating the Margin between iw and sw
margin = int((sw_size - iw_size) / 2)
# Number of iw and sw in each direction
iw_no_y = int(2*np.floor((height - 1 - iw_size) / (iw_size - 1)))
iw_no_x = int(2*np.floor((width - 1 - iw_size) / (iw_size - 1)))

iw_no_x = int(width/(int(iw_size - iw_size * overlap)+1))+1
# Initializing Displacement field
# Displacement in x and y direction
vec_x_ = []
vec_x_total = []
vec_y_ = []
vec_y_total = []
vec_ = []
vec_total = []
# Initializing the Correlation Matrix
rij = []
rj = []

iw_right_bound = 0
iw_top_bound = 0
iw_index = 0
while iw_right_bound <= width:
    # Moving in the X direction
    # Calculation of the boundaries of the interrogation window
    iw_left_bound = int(iw_index * ((iw_size - 1) * (1-overlap)))
    iw_right_bound = iw_left_bound + iw_size
    # Calculation of the boundaries of the search window
    sw_left_bound = max(0, iw_left_bound - margin)
    sw_d_diff = max(0, iw_left_bound - margin) - (iw_left_bound - margin)
    sw_right_bound = min(width - 1, iw_right_bound + margin)
    
    iw_y_index = 0
    iw_top_bound = 0
    
    while iw_top_bound <= height:
        # Moving in the Y direction
        # Calculation of the boundaries of the interrogation window
        iw_bottom_bound = int(iw_y_index * ((iw_size - 1) * (1-overlap)))
        iw_top_bound = iw_bottom_bound + iw_size
        # Calculation of the boundaries of the search window
        sw_bottom_bound = max(0, iw_bottom_bound - margin)
        sw_l_diff = max(0, iw_bottom_bound - margin) - (iw_bottom_bound - margin)
        sw_top_bound = min(height - 1, iw_top_bound + margin)
        # Getting an IW from the first image
        c1 = img_1[..., iw_bottom_bound:iw_bottom_bound + iw_size, iw_left_bound:iw_left_bound + iw_size]
        center_pixel_c1_in_image_1 = torch.Tensor([iw_bottom_bound + iw_size // 2, iw_left_bound + iw_size // 2])
        # Getting the SW from the second image
        sw2 = img_2[..., sw_bottom_bound:sw_top_bound, sw_left_bound:sw_right_bound]
        # Calculating the Correlation 
        R_torch = utils.torch_corr(c1, sw2)
        # Getting the Maximum Correlation
        R_max = R_torch.max()
        rj.append(R_max)
        # Calculating the displacement of the maximum correlation
        if R_max >= config["r_limit"]:
            # dummy variable to get the coordinate of the maximum correlation
            dum = np.unravel_index(np.argmax(R_torch), R_torch.shape)
            # Finding out where the max corr happens in the second image
            center_of_max_R_in_image_2 = np.array([sw_bottom_bound + dum[2], sw_left_bound + dum[1]])
            # Calculating the displacement of the center of the IW from the first image and the place of max_corr 
            # from the SW of the second image.
            y_displacement = center_of_max_R_in_image_2[1] - center_pixel_c1_in_image_1[1] + utils.subpix(R_torch[0], 'y', dum)
            vec_y_.append(y_displacement)
            x_displacement = center_of_max_R_in_image_2[0] - center_pixel_c1_in_image_1[0] + utils.subpix(R_torch[0], 'x', dum)
            vec_x_.append(x_displacement)
            total_displacement = (x_displacement**2 + y_displacement**2)**0.5
            vec_.append(total_displacement)
        else:
            vec_x_.append(0.0)
            vec_y_.append(0.0)
            vec_.append(0.0)
        iw_y_index += 1
    iw_index += 1
    rij.append(rj)
    rj = []
    vec_x_total.append(vec_x_)
    vec_x_ = []
    vec_y_total.append(vec_y_)
    vec_y_ = []
    vec_total.append(vec_)
    vec_ = []
    print(f"IW No. {iw_index}")

rij = np.array(rij).T
vec = np.array(vec_total).T
vec_x = np.array(vec_x_total).T
vec_y = np.array(vec_y_total).T

# Filtering out the outliers with the median
vecx, vecy, vec, i_disorder, i_cor_done = utils.fixer(vec_x, vec_y, vec, rij, config["r_limit"], config["i_fix"])

# Plotting Parameters
output_dir = config["output_dir"]
X, Y = np.meshgrid(np.arange(0.5*iw_size, 0.5*iw_size*(rij.shape[1]+1), 0.5*iw_size), 
                   np.arange(0.5*iw_size, 0.5*iw_size*(rij.shape[0]+1), 0.5*iw_size))

# Spatial and Temporal scaling
X *= l_scale
Y *= l_scale
vecx *= (l_scale / t_scale)
vecy *= (l_scale / t_scale)
vec *= (l_scale / t_scale)

# Skipping boundary Interrogation Windows
sk = 1 
X = X[sk:-sk, sk:-sk]
Y = Y[sk:-sk, sk:-sk]
vecx = vecx[sk:-sk, sk:-sk]
vecy = vecy[sk:-sk, sk:-sk]
vec = vec[sk:-sk, sk:-sk]
rij = rij[sk:-sk, sk:-sk]

np.savez(f"{output_dir}/results.npz", vecx = vecx, vecy = vecy, vec = vec, rij = rij)

fig, ax = plt.subplots(figsize=(10,8))
q = ax.quiver(X, Y, vecx, vecy, units='width')
plt.savefig(f"{output_dir}/figs/velocity_field.png", dpi=300)

fig, ax = plt.subplots(figsize=(10,8))
plt.contourf(X[0],np.transpose(Y)[0],rij,cmap='jet',levels=np.arange(rij.min(),min(rij.max()+0.1,1.0),0.01))
plt.colorbar(label='R')
plt.savefig(f"{output_dir}/figs/correlation_field.png", dpi=300)

fig, ax = plt.subplots(figsize=(10,8))
plt.streamplot(X, Y, vecx, vecy, density = 3, linewidth = 0.8, color = vec)
plt.savefig(f"{output_dir}/figs/streamline.png", dpi=300)
plt.colorbar(label='velocity')
plt.show()