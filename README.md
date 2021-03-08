# centertrack_wami
This repository holds the code for our work on Wide area motion images. We choose to work on [WPAFB dataset](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009) available at  and compare our results with [paper](https://arxiv.org/abs/1911.01727). 

## Dataset Preprocessing
We choose to work only on r1 resolution images of size 13312 X 10752  and download the training and test portions of [WPAFB dataset](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009). Run wapfb_save.py file to convert NITF images into PNG files. Then we extract Regions of Interest (ROI) specified by [paper](https://arxiv.org/abs/1911.01727) with location and size specified in their [MATLAB repository](https://github.com/zhouyifan233/MovingObjDetector-WAMI.matlab) using **run_area_of_interest_test_set.m** file located in the testing_code folder. This should generate registered AOI sequences and annotations for moving vehicles. Then we generate small crops of fixed size 960 X 544 using **crop_AOI.py** file which will generate video sequences.

## Training Code
Training is done using CenterTrack and most of the training code is taken from [official CenterTrack repository](https://github.com/xingyizhou/CenterTrack). CenterTrack expects annotations in COCO format. Hence, convert the annotations of moving vehicles into COCO format using **convert_wpafb_to_coco.py** file.
Run main.py using the below command. **Replace train_LR.json with your COCO format annotation file and train_LR sequences path with your sequences of crops.**
~~~
python test.py tracking --exp_id wpafb --dataset wpafb --custom_dataset_ann_path ../data/wpafb/annotations/train_LR.json --custom_dataset_img_path ../data/wpafb/train_LR/sequences/ --input_h 544 --input_w 960 --num_classes 2 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../exp/tracking/wpafb_LR/model_epoch75.pth  --gpus 0
~~~~~~
