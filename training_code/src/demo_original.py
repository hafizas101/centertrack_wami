from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os, time, glob
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector

output_video_path = 'output.mp4'
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
class_name = ['vehicle']
obj_category = ['1']
class_colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)
  start = time.time()
  annots = os.path.join(opt.demo, 'annotations_moving')
  seqs = os.path.join(opt.demo, 'sequences')
  
  if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    # demo on video stream
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  else:
    is_video = False
    # Demo on images sequences
    vids = sorted(os.listdir(seqs))
    for i, vid_num in enumerate(vids):
      vid_path = os.path.join(seqs, vid_num)
      print(vid_path)
      image_names = []
      ls = os.listdir(vid_path)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(vid_path, file_name))
      
      cnt = 0
      results = []
      labelled = []
      det = os.path.join(opt.demo, 'ct_locations/'+vid_num+'.txt')
      # f = open(det, 'w')
      while True:
          if is_video:
            _, img = cam.read()
            if img is None:
              save_and_exit(opt, results )
          else:
            if cnt < len(image_names):
              # print(image_names[cnt])
              img = cv2.imread(image_names[cnt])
              fr = image_names[cnt].split('/')[-1].split('.')[0]
              cnt = cnt + 1
            else:
              break
           
          # resize the original video for saving video results
          if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))
    
            # skip the first X frames of the video
          if cnt < opt.skip_first:
            continue
          
          ret = detector.run(img)
          results.append(ret['results'])
    
          img2 = img.copy()
          # print(len(ret['results']))
          
          if len(ret['results']) != 0:
              for j, ll in enumerate(ret['results']):
                  score = ll['score']
                  trd = ll['tracking_id']
                  bb = ll['bbox']
                  # center = (int(bb[0]), int(bb[1]))
                  center = (int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2))
                  # f.write(fr+','+str(trd)+','+str(center[0])+','+str(center[1])+',11,11,1,1\n')
                  cv2.circle(img2, (center[0], center[1]), 5, (255,0,0), -1)
                  # cv2.imshow("Image", img2)
                  # cv2.waitKey(0)
      # f.close()
    cv2.destroyAllWindows()
              
  #          f.close()


def save_and_exit(opt, out=None, results=None, out_name=''):
  cv2.destroyAllWindows()
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
