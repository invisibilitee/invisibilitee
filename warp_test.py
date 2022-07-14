
import torch
import torchgeometry as tgm

import cv2

# read the image with OpenCV
image = cv2.imread('lena416.png')[..., (2,1,0)]
print(image.shape)
'''
img = tgm.image_to_tensor(image)
img = torch.unsqueeze(img.float(), dim=0)  # BxCxHxW

# create transformation (rotation)
alpha = 45.0  # in degrees
angle = torch.ones(1) * alpha

# define the rotation center
center = torch.ones(1, 2)
center[..., 0] = img.shape[3] / 2  # x
center[..., 1] = img.shape[2] / 2  # y

# define the scale factor
scale = torch.ones(1)

# compute the transformation matrix
M = tgm.get_rotation_matrix2d(center, angle, scale)

# apply the transformation to original image
_, _, h, w = img.shape
img_warped = tgm.warp_affine(img, M, dsize=(h, w))

# convert back to numpy
image_warped = tgm.tensor_to_image(img_warped.byte())
'''

'''
method 2
'''
points_src = torch.FloatTensor([[
    [125, 150], [562, 40], [562, 282], [54, 328],
]])

# the destination points are the image vertexes
h, w = 64, 128  # destination size
points_dst = torch.FloatTensor([[
    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
]])

# compute perspective transform
M = tgm.get_perspective_transform(points_src, points_dst)

# warp the original image by the found transform
img_warp = tgm.warp_perspective(img, M, dsize=(h, w))

# convert back to numpy
image_warp = tgm.tensor_to_image(img_warp.byte())

# draw points into original image
for i in range(4):
    center = tuple(points_src[0, i].long().numpy())
    image = cv2.circle(image.copy(), center, 5, (0, 255, 0), -1)
