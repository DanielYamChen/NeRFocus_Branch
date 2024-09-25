import cv2
import numpy as np
import os

# 3, 7, 15, 31, 51, 71, 101
kernel_size = 51
data_in_path = './nerf_synthetic/materials/train'
data_out_path = './nerf_synthetic/materials_Gaussblur/' + str(kernel_size) + '/train/'

if __name__ == "__main__":
    imgs_T = os.listdir(data_in_path)
    imgs_T.sort()
    for image_name in imgs_T:
        if image_name == '.ipynb_checkpoints':
            continue
        img_T = cv2.imread(os.path.join(data_in_path, image_name))
        dst = cv2.GaussianBlur(img_T, (kernel_size, kernel_size), 0) 
        assert(cv2.imwrite(os.path.join(data_out_path, image_name), dst))
        # cv2.waitKey()
