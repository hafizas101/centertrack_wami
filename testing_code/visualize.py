# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Feb  1 12:23:35 2021

# @author: casper
# """

import cv2, glob

AOI_ID = '01'

img_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/sequences/*'))
gt_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/annotations_moving/*'))
ct_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/ct_locations_prof/*'))

if (len(img_files) == len(gt_files)) and (len(gt_files) == len(ct_files)):
    for i, name in enumerate(img_files):
        img = cv2.imread(name, 0)
        f1 = open(gt_files[i], 'r')
        data1 = f1.read()
        lines1= data1.split("\n")[:-1]
        
        f2 = open(ct_files[i], 'r')
        data2 = f2.read()
        lines2= data2.split("\n")[:-1]
        
        for j,l1 in enumerate(lines1):
            words = l1.split(',')
            left = int(words[0])
            top = int(words[1])
            cv2.circle(img, (left, top), 5, (0,0,0), -1)
           

        for j,l2 in enumerate(lines2):
            words = l2.split(',')
            left = int(words[0])
            top = int(words[1])
            cv2.circle(img, (left, top), 5, (255,255,0), -1)



        cv2.imshow("Image", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# gt_file = '/home/casper/Desktop/CenterTrack/data/wpafb/train_LR/annotations_moving/Vid0.txt'
# images = sorted(glob.glob('/home/casper/Desktop/CenterTrack/data/wpafb/train_LR/sequences/Vid0/*'))
# tr_id = '213'

# f = open(gt_file, "r")
# data = f.read()
# lines = data.split("\n")[:-1]

# for i, name in enumerate(images):
#     img = cv2.imread(name, 0)
#     img2 = img.copy()
#     img_id = name.split('/')[-1].split('.')[0]
    
#     for j,l in enumerate(lines):
#         words = l.split(',')
#         immm = words[0]
#         if img_id == immm:
#             cv2.circle(img2, (int(float(words[2])), int(float(words[3]))), 5, (0,0,0), -1)
#             cv2.imshow("Image", img2)
#             cv2.waitKey(0)
# cv2.destroyAllWindows()




# gt_file = 'AOI/AOI_01/train_annotations_moving/0105.txt'
# img = cv2.imread('AOI/AOI_01/train_sequences/0105.png', 1)

# f = open(gt_file, "r")
# data = f.read()
# lines = data.split("\n")[:-1]
    
# for j,l in enumerate(lines):
#     words = l.split(',')
#     cv2.circle(img, (int(float(words[0])), int(float(words[1]))), 5, (0,0,0), -1)
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# tr_id = '213'
# AOI_ID = '01'
# img_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/train_sequences/*'))
# gt_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/train_annotations_moving/*'))
# # ct_files = sorted(glob.glob('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_'+str(AOI_ID)+'/ct_locations_prof/*'))

# if (len(img_files) == len(gt_files)) and (len(gt_files)):
#     for i, name in enumerate(img_files):
#         img = cv2.imread(name, 0)
#         f1 = open(gt_files[i], 'r')
#         data1 = f1.read()
#         lines1= data1.split("\n")[:-1]
        
#         for j,l1 in enumerate(lines1):
#             words = l1.split(',')
#             if words[2] == tr_id:
#                 left = int(words[0])
#                 top = int(words[1])
#                 cv2.circle(img, (left, top), 5, (0,0,0), -1)
#                 cv2.imshow("Image", img)
#                 cv2.waitKey(0)
#     cv2.destroyAllWindows()



