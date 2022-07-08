# -*- coding: utf-8 -*-
"""
Particle Image Velocimetry Using 
pyTorch implemented Normalized Cross Correlation
@author: Erfan Hamdi Jan 2022
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch
from torchvision import transforms
import utils
import warnings
warnings.filterwarnings("ignore")

# Image transformations using pytorch
image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])

# Input images
img_1 = Image.open('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/1.jpg')
img_1 = image_transforms(img_1)

img_2 = Image.open('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/2.jpg')
img_2 = image_transforms(img_2)

# Output Params
frame = 2

# Algorithm Params
# Maximum Fixing Iterations
i_fix = 500
# R correlation threshold
r_limit = 0.5
# Spatial Scale [m/pixel]
l_scale = 1.0
# Temporal Scale 1/frame_rate [s/frame]
t_scale = 1.0
# Interrodation Windows Sizes (pixel)
iw_size = 51 
# Search Windows Sizes (sw > iw) (pixel)
sw_size = 81 

batch, height, width = img_1.shape

# Changing the shape of windows from even to odd
iw_size = int( 2 * np.floor( (iw_size + 1) / 2 ) - 1 )
sw_size = int( 2 * np.floor( (sw_size + 1) / 2 ) - 1 )
# Calculating the Margin between iw and sw
margin = int((sw_size - iw_size) / 2)
# Number of iw and sw in each direction
iw_no_y = int(2*np.floor((height - 1 - iw_size) / (iw_size - 1)))
iw_no_x = int(2*np.floor((width - 1 - iw_size) / (iw_size - 1)))

overlap = 0.5
iw_no_x = int(width/(int(iw_size - iw_size * overlap)+1))+1
# Initializing Displacement field
# Displacement in x and y direction
vecx = np.zeros((iw_no_y, iw_no_x)) # x-Displacement
vecy = np.zeros((iw_no_y, iw_no_x)) # y-Displacement
vec = np.zeros((iw_no_y, iw_no_x)) # Magnitude
vec_x_ = []
vec_x_total = []
vec_y_ = []
vec_y_total = []
vec_ = []
vec_total = []
# Initializing the Correlation Matrix
rij = np.zeros((iw_no_y, iw_no_x)) # Correlation coeff.
rij2 = []
rj = []
iw_right_bound = 0
# while iw_right_bound
for iw_x in tqdm(range(iw_no_x)):
    iw_left_bound = int(iw_x * ((iw_size - 1) * (1-overlap)))
    iw_right_bound = iw_left_bound + iw_size

    sw_left_bound = max(0, iw_left_bound - margin)
    sw_d_diff = max(0, iw_left_bound - margin) - (iw_left_bound - margin)
    sw_right_bound = min(width - 1, iw_right_bound + margin)

    for iw_y in range(iw_no_y):
        iw_bottom_bound = int(iw_y * ((iw_size - 1) * (1-overlap)))
        iw_top_bound = iw_bottom_bound + iw_size
        
        sw_bottom_bound = max(0, iw_bottom_bound - margin)
        sw_l_diff = max(0, iw_bottom_bound - margin) - (iw_bottom_bound - margin)
        sw_top_bound = min(height - 1, iw_top_bound + margin)
        
        # R = np.zeros((sw_size - iw_size + 1, sw_size - iw_size + 1)) - 1 # Correlation Matrix
        c1 = img_1[..., iw_bottom_bound:iw_bottom_bound + iw_size, iw_left_bound:iw_left_bound + iw_size]# IW from 1st image
        center_pixel_c1_in_image_1 = torch.Tensor([iw_bottom_bound + iw_size // 2, iw_left_bound + iw_size // 2])
        R_torch = utils.torch_corr(c1, img_2[..., sw_bottom_bound:sw_top_bound, sw_left_bound:sw_right_bound])
        R_max = R_torch.max()
        rj.append(R_max)
        rij[iw_y, iw_x] = R_torch.max()
        # if rij[iw_y, iw_x] >= r_limit:
        if R_max >= r_limit: 
            # dum=np.floor(np.argmax(R)/R.shape[0])
            dum = np.unravel_index(np.argmax(R_torch), R_torch.shape)
            center_of_max_R_in_image_2 = np.array([sw_bottom_bound + dum[2], sw_left_bound + dum[1]])
            vecy[iw_y, iw_x] = center_of_max_R_in_image_2[1] - center_pixel_c1_in_image_1[1] + utils.subpix(R_torch[0], 'y', dum)
            y_displacement = center_of_max_R_in_image_2[1] - center_pixel_c1_in_image_1[1] + utils.subpix(R_torch[0], 'y', dum)
            vec_y_.append(y_displacement)
            vecx[iw_y, iw_x] = center_of_max_R_in_image_2[0] - center_pixel_c1_in_image_1[0] + utils.subpix(R_torch[0], 'x', dum)
            x_displacement = center_of_max_R_in_image_2[0] - center_pixel_c1_in_image_1[0] + utils.subpix(R_torch[0], 'x', dum)
            vec_x_.append(x_displacement)
            # vecy[i,j]=dum-(margin-sw_l_diff)+subpix(R,'y')

            # vecx[i,j]=np.argmax(R)-dum*R.shape[0]-(margin-sw_left_bound_diff)+subpix(R,'x')
            total_displacement = (x_displacement**2 + y_displacement**2)**0.5
            vec[iw_y, iw_x] = np.sqrt(vecx[iw_y, iw_x] * vecx[iw_y, iw_x] + vecy[iw_y, iw_x] * vecy[iw_y, iw_x])
            vec_.append(total_displacement)

        else:
            vecx[iw_y, iw_x]=0.0
            vec_x_.append(0.0)
            vec_y_.append(0.0)
            vec_.append(0.0)
            vecy[iw_y, iw_x]=0.0
            vec[iw_y, iw_x]=0.0
    rij2.append(rj)
    rj = []
    vec_x_total.append(vec_x_)
    vec_x_ = []
    vec_y_total.append(vec_y_)
    vec_y_ = []
    vec_total.append(vec_)
    vec_ = []
rij = np.array(rij2).T
vec = np.array(vec_total).T
vec_x = np.array(vec_x_total).T
vec_y = np.array(vec_y_total).T
vecx, vecy, vec, i_disorder, i_cor_done = utils.fixer(vecx, vecy, vec, rij, r_limit, i_fix)

X, Y = np.meshgrid(np.arange(0.5*iw_size, 0.5*iw_size*(iw_no_x+1), 0.5*iw_size), 
                   np.arange(0.5*iw_size, 0.5*iw_size*(iw_no_y+1), 0.5*iw_size))
X *= l_scale
Y *= l_scale

vecx *= (l_scale / t_scale)
vecy *= (l_scale / t_scale)
vec *= (l_scale / t_scale)

np.savez('results.npz', vecx = vecx, vecy = vecy, vec = vec, rij = rij)

# res=np.load('results.npz'); vecx=res['vecx']; vecy=res['vecy']; vec=res['vec']; rij=res['rij']; # Load saved data
# fig, ax = plt.subplots(figsize=(8,8*ia/ja))
# q = ax.quiver(X, Y, vecx, vecy,units='width')
# plt.savefig('velocity_jet.png', dpi=300)
# plt.show()

fig, ax = plt.subplots(figsize = (8, 8 * height / width))
plt.contourf(X[0], np.transpose(Y)[0], rij, cmap='jet', levels=np.arange(rij.min(), min(rij.max() + 0.1, 1.0), 0.01))
plt.colorbar(label='R')
# plt.savefig(f'/Users/venus/Erfan/sharifCourses/optic/piv/Report/figs/mixing/r_corr_jet_t{frame}_{iw}_{sw}.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8 * height / width))
plt.streamplot(X, Y, vecx, vecy, density = 3, linewidth = 0.8, color = vec)
# plt.savefig(f'/Users/venus/Erfan/sharifCourses/optic/piv/Report/figs/mixing/velocity_jet_t{frame}_{iw}_{sw}.png', dpi=300)
plt.show()