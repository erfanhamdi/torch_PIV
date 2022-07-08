# -*- coding: utf-8 -*-
"""
Particle Image Velocimetry Using 
pyTorch implemented Normalized Cross Correlation
@author: Erfan Hamdi Jan 2022
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch
from torchvision import transforms
from NCC import NCC
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

# img_1 = (np.flip(cv2.imread('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/1.jpg', 0),0)).astype('float32') # Read Grayscale
# img_2 = (np.flip(cv2.imread('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/2.jpg', 0),0)).astype('float32')
# Output Params
frame = 2
# img_1 = (np.flip(cv2.imread('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/a1.jpg', 0),0)).astype('float32') # Read Grayscale
# img_2 = (np.flip(cv2.imread('/Users/venus/Erfan/sharifCourses/optic/piv/PIV/Example/a2.jpg', 0),0)).astype('float32') # Read Grayscalei_fix=500     # Number of maximum correction cycles

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
iw=51 
# Search Windows Sizes (sw > iw) (pixel)
sw=81 

batch, ia, ja = img_1.shape
# ia, ja = img_1.shape
iw=int(2*np.floor((iw+1)/2)-1) # Even->Odd
sw=int(2*np.floor((sw+1)/2)-1)
margin=int((sw-iw)/2)
im=int(2*np.floor((ia-1-iw)/(iw-1))) # Number of I.W.s in x direction
jm=int(2*np.floor((ja-1-iw)/(iw-1))) # Number of I.W.s in y direction

vecx=np.zeros((im,jm)) # x-Displacement
vecy=np.zeros((im,jm)) # y-Displacement
vec=np.zeros((im,jm)) # Magnitude
rij=np.zeros((im,jm)) # Correlation coeff.

for j in tqdm(range(jm)):
    j_d=int(j*(iw-1)/2) # Bottom bound
    j_u=j_d+iw          # Top bound
    sw_d=max(0,j_d-margin) # First Row
    sw_d_diff=max(0,j_d-margin)-(j_d-margin)
    sw_u=min(ja-1,j_u+margin) # Last Row
    
    for i in range(im):
        i_l=int(i*(iw-1)/2) # Left bound
        i_r=i_l+iw          # Right bound
        sw_l=max(0,i_l-margin) # First column
        sw_l_diff=max(0,i_l-margin)-(i_l-margin)
        sw_r=min(ia-1,i_r+margin) # Last column
        
        R=np.zeros((sw-iw+1,sw-iw+1))-1 # Correlation Matrix
        # c1=np.array(img_1[i_l:i_l+iw,j_d:j_d+iw]) # IW from 1st image
        c1=img_1[..., i_l:i_l+iw,j_d:j_d+iw]# IW from 1st image
        center_pixel_c1_in_image_1=torch.Tensor([i_l+iw//2,j_d+iw//2])
        R_torch = utils.torch_corr(c1, img_2[..., sw_l:sw_r,sw_d:sw_u])
        # for jj in range(sw_d,sw_u+1-iw):
        #     for ii in range(sw_l,sw_r+1-iw):
        #         c2=np.array(img_2[ii:ii+iw,jj:jj+iw]) # IW from 2nd image
        #         R[ii-sw_l,jj-sw_d]=corr2(c1,c2)
        # rij[i,j]=R.max()
        rij[i,j]=R_torch.max()
        if rij[i,j]>=r_limit:
            # dum=np.floor(np.argmax(R)/R.shape[0])
            dum = np.unravel_index(np.argmax(R_torch), R_torch.shape)
            center_of_max_R_in_image_2 = np.array([sw_l+dum[2],sw_d+dum[1]])
            vecy[i, j] = center_of_max_R_in_image_2[1] - center_pixel_c1_in_image_1[1] + utils.subpix(R_torch[0], 'y', dum)
            vecx[i, j] = center_of_max_R_in_image_2[0] - center_pixel_c1_in_image_1[0] + utils.subpix(R_torch[0], 'x', dum)
            # vecy[i,j]=dum-(margin-sw_l_diff)+subpix(R,'y')

            # vecx[i,j]=np.argmax(R)-dum*R.shape[0]-(margin-sw_d_diff)+subpix(R,'x')
            vec[i,j]=np.sqrt(vecx[i,j]*vecx[i,j]+vecy[i,j]*vecy[i,j])
        else:
            vecx[i,j]=0.0;vecy[i,j]=0.0;vec[i,j]=0.0
        
vecx,vecy,vec,i_disorder,i_cor_done=utils.fixer(vecx,vecy,vec,rij,r_limit,i_fix)


X, Y = np.meshgrid(np.arange(0.5*iw, 0.5*iw*(jm+1), 0.5*iw), 
                   np.arange(0.5*iw, 0.5*iw*(im+1), 0.5*iw))
X*=l_scale
Y*=l_scale

vecx*=(l_scale/t_scale);vecy*=(l_scale/t_scale);vec*=(l_scale/t_scale);




np.savez('results.npz', vecx=vecx, vecy=vecy, vec=vec, rij=rij)
# res=np.load('results.npz'); vecx=res['vecx']; vecy=res['vecy']; vec=res['vec']; rij=res['rij']; # Load saved data

# fig, ax = plt.subplots(figsize=(8,8*ia/ja))
# q = ax.quiver(X, Y, vecx, vecy,units='width')
# plt.savefig('velocity_jet.png', dpi=300)
# plt.show()


fig, ax = plt.subplots(figsize=(8,8*ia/ja))
plt.contourf(X[0],np.transpose(Y)[0],rij,cmap='jet',levels=np.arange(rij.min(),min(rij.max()+0.1,1.0),0.01))
plt.colorbar(label='R')
# plt.savefig(f'/Users/venus/Erfan/sharifCourses/optic/piv/Report/figs/mixing/r_corr_jet_t{frame}_{iw}_{sw}.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8,8*ia/ja))
plt.streamplot(X, Y, vecx, vecy,density=3,linewidth=0.8,color=vec)
# plt.savefig(f'/Users/venus/Erfan/sharifCourses/optic/piv/Report/figs/mixing/velocity_jet_t{frame}_{iw}_{sw}.png', dpi=300)
plt.show()
