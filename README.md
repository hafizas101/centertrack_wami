# centertrack_wami
This repository holds the code for our work on Wide area motion images. We choose to work on [WPAFB dataset](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009) available at  and compare our results with [paper](https://arxiv.org/abs/1911.01727). 

## Dataset Preprocessing
We choose to work only on r1 resolution images of size 13312 X 10752  and download the training and test portions of [WPAFB dataset](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009). Run wapfb_save.py file to convert NITF images into PNG files. Then we extract Regions of Interest (ROI) specified by [paper](https://arxiv.org/abs/1911.01727) with location and size specified in their [MATLAB repository](https://github.com/zhouyifan233/MovingObjDetector-WAMI.matlab) using **run_area_of_interest_test_set.m** file located in the testing_code folder. This should generate registered AOI sequences and annotations for moving vehicles. Then we generate small crops of fixed size 960 X 544 using **crop_AOI.py** file which will generate video sequences.

## Training Code
Training is done using CenterTrack and most of the training code is taken from [official CenterTrack repository](https://github.com/xingyizhou/CenterTrack). CenterTrack expects annotations in COCO format. Hence, convert the annotations of moving vehicles into COCO format using **convert_wpafb_to_coco.py** file.
Run main.py using the below command. **Replace train_LR.json with your COCO format annotation file and train_LR sequences path with your sequences of crops.**
~~~
python main.py tracking --exp_id wpafb --dataset wpafb --custom_dataset_ann_path ../data/wpafb/annotations/train_LR.json --custom_dataset_img_path ../data/wpafb/train_LR/sequences/ --input_h 544 --input_w 960 --num_classes 2 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --gpus 0 --num_epochs 100 --save_point 20,40,60,80
~~~

This will save trained models in exp/tracking/wpafb/ directory at 20,40,60,80 epochs and also the code saves a model_last.pth file after every epoch.

## Testing Code
Testing is done using **demo_AOI.py** and **demo.py** files located in training_code/CenterTrack/src to generate the detection .txt files. Make sure to put appropritate test image width and height. Also, replace trained model path and demo sequence paths. Run the file using the command:
~~~
python demo_AOI.py tracking --num_classes 2 --demo /home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_01 --load_model ../exp/tracking/crops_vids_1/model_last.pth --input_h 1216 --input_w 1216 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5
~~~
Final testing results of Avg. Precision, Avg. Recall and F1 score are generated using **evaluate_python_results.m**. We choose these three evaluation metrics to make a fair comparison with the results of [WAMI paper](https://arxiv.org/abs/1911.01727).

## Results
We compared our CenterTrack results in terms of accuracy and speed with their results. Comparison of accuracy is done with their [MATLAB repository](https://github.com/zhouyifan233/MovingObjDetector-WAMI.matlab) and speed with their [Python repository](https://github.com/zhouyifan233/MovingObjDetector-WAMI.python). The results are shown in the below figures.
<p align="center">
  <img width="600" height="350" src="https://github.com/hafizas101/centertrack_wami/blob/master/result_1.png">
</p>

<p align="center">
  <img width="600" height="370" src="https://github.com/hafizas101/centertrack_wami/blob/master/result_2.png">
</p>

Results demonstrate that CenterTrack trained model achieves similar results at almost 10 times faster.
