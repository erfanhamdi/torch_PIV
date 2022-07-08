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
i_fix=500
# R correlation threshold
r_limit=0.5
# Spatial Scale [m/pixel]
l_scale=1.0
# Temporal Scale 1/frame_rate [s/frame]
t_scale=1.0
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
iw_no_y = int(2*np.floor((height-1-iw_size)/(iw_size-1)))
iw_no_x = int(2*np.floor((width-1-iw_size)/(iw_size-1)))

# Initializing Displacement field
# Displacement in x and y direction
vecx = np.zeros((iw_no_y,iw_no_x)) # x-Displacement
vecy = np.zeros((iw_no_y,iw_no_x)) # y-Displacement
vec = np.zeros((iw_no_y,iw_no_x)) # Magnitude

# Initializing the Correlation Matrix
rij = np.zeros((iw_no_y,iw_no_x)) # Correlation coeff.

for iw_x in tqdm(range(iw_no_x)):
    j_d = int(iw_x * (iw_size - 1) / 2) # Bottom bound
    j_u = j_d + iw_size          # Top bound
    sw_d = max(0, j_d - margin) # First Row
    sw_d_diff = max(0, j_d - margin) - (j_d - margin)
    sw_u = min(width - 1, j_u + margin) # Last Row

    for iw_y in range(iw_no_y):
        i_l = int(iw_y * (iw_size - 1) / 2) # Left bound
        i_r = i_l + iw_size          # Right bound
        sw_l = max(0, i_l - margin) # First column
        sw_l_diff = max(0, i_l - margin) - (i_l - margin)
        sw_r = min(height - 1, i_r + margin) # Last column
        R = np.zeros((sw_size - iw_size + 1, sw_size - iw_size + 1)) - 1 # Correlation Matrix
        c1 = img_1[..., i_l:i_l + iw_size, j_d:j_d + iw_size]# IW from 1st image
        center_pixel_c1_in_image_1 = torch.Tensor([i_l + iw_size // 2, j_d + iw_size // 2])
        R_torch = utils.torch_corr(c1, img_2[..., sw_l:sw_r, sw_d:sw_u])
        rij[iw_y, iw_x] = R_torch.max()
        if rij[iw_y, iw_x] >= r_limit:
            # dum=np.floor(np.argmax(R)/R.shape[0])
            dum = np.unravel_index(np.argmax(R_torch), R_torch.shape)
            center_of_max_R_in_image_2 = np.array([sw_l + dum[2], sw_d + dum[1]])
            vecy[iw_y, iw_x] = center_of_max_R_in_image_2[1] - center_pixel_c1_in_image_1[1] + utils.subpix(R_torch[0], 'y', dum)
            vecx[iw_y, iw_x] = center_of_max_R_in_image_2[0] - center_pixel_c1_in_image_1[0] + utils.subpix(R_torch[0], 'x', dum)
            # vecy[i,j]=dum-(margin-sw_l_diff)+subpix(R,'y')

            # vecx[i,j]=np.argmax(R)-dum*R.shape[0]-(margin-sw_d_diff)+subpix(R,'x')
            vec[iw_y, iw_x] = np.sqrt(vecx[iw_y, iw_x] * vecx[iw_y, iw_x] + vecy[iw_y, iw_x] * vecy[iw_y, iw_x])
        else:
            vecx[iw_y, iw_x]=0.0
            vecy[iw_y, iw_x]=0.0
            vec[iw_y, iw_x]=0.0
        
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