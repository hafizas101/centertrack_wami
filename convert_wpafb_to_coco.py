import os
import numpy as np
import json
import cv2
import pdb

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = './'
OUT_PATH = DATA_PATH + 'annotations/'
#SPLITS = ['train', 'val', 'test-dev']
SPLITS = ['train_crops_vids_1']
img_len = 3

# if 'train' in SPLITS[0]:
#     img_len = 3
# else:
#     img_len = 4

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + (split) + '/sequences'
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 0, 'name': 'ignore'}, 
                          {'id': 1, 'name': 'vehicle'}, 
                          ], 'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    #pdb.set_trace()
    for seq in sorted(seqs):
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path 
      ann_path = DATA_PATH +  (split) + '/annotations/'+seq+'.txt'
      images = sorted(os.listdir(img_path))
      num_images = len([image for image in images if 'png' in image])
      image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/{}'.format(seq, images[i]),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test_challenge':
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          #HFA -DİKKAT!  
          frame_id = ((int(anns[i][0]) - int(images[0][0:img_len]))//1) + 1
          #frame_id = images.index(str(int(anns[i][0]))+'.png') +1
          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][6])
          ann_cnt += 1
          
          # Edit for center compensation
          anns[i][2] = abs(anns[i][2] - 7)
          anns[i][3] = abs(anns[i][3] - 7)
          
          if anns[i][2] < 0:
              anns[i][2] = 0
          if anns[i][3] < 0:
              anns[i][3] = 0
          
          
          
          ann = {'id': ann_cnt,
                 'category_id': cat_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'conf': float(anns[i][7])}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
        

