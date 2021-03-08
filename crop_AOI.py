#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:51:14 2021
@author: casper
"""
import glob, cv2, os
import numpy as np
import pandas as pd

AOI_ID = '01'

file_names = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/train_sequences/*'))
annot_names = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/train_annotations_moving/*'))
out_seq = '/home/casper/Desktop/MovingObjDetector-WAMI.matlab/cropped/AOI_'+str(AOI_ID)+'/train_sequences/'
out_annot = '/home/casper/Desktop/MovingObjDetector-WAMI.matlab/cropped/AOI_'+str(AOI_ID)+'/train_annotations_moving'
cr_sz = [960,544]

img = cv2.imread(file_names[0], 0)
[m,n] = img.shape
w_crops = int(n/cr_sz[0]) + 1
h_crops = int(m/cr_sz[1]) + 1
num_vids = w_crops*h_crops
f_annots = []
vid_crops_list = []
vid_folders = []
fs = []

for aa in range(num_vids):
    vid_id = str(aa).zfill(3)
    vid_folder = os.path.join(out_seq, 'Vid'+vid_id)
    vid_folders.append(vid_folder)
    if not os.path.exists(vid_folder):
        os.mkdir(vid_folder)
    fs.append(open(os.path.join(out_annot, 'Vid'+vid_id+'.txt'), 'w'))
    
    

for i, name in enumerate(file_names):
    img_id = name.split('/')[-1].split('.')[0]
    print(img_id)
    img = cv2.imread(name, 0)
    data = pd.read_csv(annot_names[i], delimiter=',')
    # print(data.shape)
    gb_left = 0
    gb_top = 0
    cnt_vid = 0
    gt_data = []
    for j in range(h_crops):
        wd = img[j*cr_sz[1]:j*cr_sz[1]+cr_sz[1], :]
        gb_top = j*cr_sz[1]
        if wd.shape[0] < cr_sz[1]:
            wd_1 = wd.copy()
            wd = np.concatenate((wd_1, np.zeros(shape=(cr_sz[1]-wd.shape[0], img.shape[1]), dtype='uint8')), axis=0)  
        for k in range(w_crops):
            crop = wd[:, k*cr_sz[0]:k*cr_sz[0]+cr_sz[0]]
            if crop.shape[1] < cr_sz[0]:
                crop = np.concatenate((crop, np.zeros(shape=(wd.shape[0], cr_sz[0]-crop.shape[1]), dtype='uint8')), axis=1)
            gb_left = k*cr_sz[0]
            # print("------------------------------------------------------------------")
            disp = crop.copy()
            cnt = 0
            for l in range(data.shape[0]):
                if data.iloc[l,0] > gb_left and data.iloc[l,1] > gb_top and data.iloc[l,0] < (gb_left+cr_sz[0]) and data.iloc[l,1] < (gb_top+cr_sz[1]):
                    # print("Data point: "+str(data.iloc[l,0]-gb_left)+", "+str(data.iloc[l,1]-gb_top))
                    cv2.circle(disp, (int(data.iloc[l,0]-gb_left), int(data.iloc[l,1]-gb_top)), 5, (0,0,0), -1)
                    tr_id = str(data.iloc[l,2])
                    fs[cnt_vid].write(img_id+','+tr_id+','+str(int(data.iloc[l,0]-gb_left-7))+','+str(int(data.iloc[l,1]-gb_top-7))+',15,15,1,1\n')
                    cnt = cnt + 1
            if cnt > 0:
                crop_id = name.split('/')[-1]
                # print(vid_folders[cnt_vid])
                crop_out = os.path.join(vid_folders[cnt_vid], crop_id)
                cv2.imwrite(crop_out, crop)
            cnt_vid = cnt_vid+1
                # cv2.imshow("Image", crop)
                # cv2.waitKey(0)
                
                
        cv2.destroyAllWindows()
for i in range(len(fs)):
    fs[i].close()
            
            
    