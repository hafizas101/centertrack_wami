# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:56:45 2020

@author: hfates
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import NITFPythonGDAL_wpafb
import pandas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time, glob

TRAIN_dir = "/home/casper/Desktop/MovingObjDetector-WAMI.python/train_NITF"
IMG_dir = "/home/casper/Desktop/MovingObjDetector-WAMI.python/train_PNG"

ls = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.python/train_NITF/*'))

vid_res = 'r1'
Image.MAX_IMAGE_PIXELS = None

# g = ls[:-85]
tic = time.time()
print("Total images: "+str(len(ls)/6))
count = 0
for file_name in sorted(ls):
    if ((file_name !='db') & (file_name[-2:]==vid_res)):
        frame_no = file_name[-15:-11]
        count = count + 1
        print(file_name)
        

        
#         img = NITFPythonGDAL_wpafb.NITF_visualize(TRAIN_dir+file_name)
#         Image.fromarray(img).save('{}/{}/{}.png'.format(IMG_dir, 'r0', frame_no))
        
        img = NITFPythonGDAL_wpafb.NITF_visualize(file_name)
        Image.fromarray(img).save('{}/{}.png'.format(IMG_dir, frame_no))
        
#         img = NITFPythonGDAL_wpafb.NITF_visualize(TRAIN_dir+file_name[:-2]+'r2')
#         Image.fromarray(img).save('{}/{}/{}.png'.format(IMG_dir, 'r2', frame_no))
        
#         img = NITFPythonGDAL_wpafb.NITF_visualize(TRAIN_dir+file_name[:-2]+'r3')
#         Image.fromarray(img).save('{}/{}/{}.png'.format(IMG_dir, 'r3', frame_no))
        
#         img = NITFPythonGDAL_wpafb.NITF_visualize(TRAIN_dir+file_name[:-2]+'r4')
#         Image.fromarray(img).save('{}/{}/{}.png'.format(IMG_dir, 'r4', frame_no))
        
#         img = NITFPythonGDAL_wpafb.NITF_visualize(TRAIN_dir+file_name[:-2]+'r5')
#         Image.fromarray(img).save('{}/{}/{}.png'.format(IMG_dir, 'r5', frame_no))

toc = time.time()
print(toc-tic)  
    