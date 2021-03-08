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
    image_names = []
    ls = os.listdir(seqs)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(seqs, file_name))
      
  cnt = 0
  labelled = []
  all_x_disp = []
  all_y_disp = []
  while True:
      if is_video:
          _, img = cam.read()
          if img is None:
            save_and_exit(opt, results )
      else:
          if cnt < len(image_names):
            
            img = cv2.imread(image_names[cnt])
            fr = image_names[cnt].split('/')[-1].split('.')[0]
            cnt = cnt + 1
          else:
            break
        
      ret = detector.run(img)
  
      img2 = img.copy()
      print(len(ret['results']))
    
  #     if len(ret['results']) != 0 and cnt > 1:
  #          # print(fr)
  #         det = os.path.join(opt.demo, 'ct_locations_prof_displacements/'+fr+'.txt')
  #         f = open(det, 'w')
  #         for j, ll in enumerate(ret['results']):
  #           score = ll['score']
  #           trd = ll['tracking_id']
  #           [disp_x, disp_y] = ll['tracking']
  #           all_x_disp.append(abs(disp_x))
  #           all_y_disp.append(abs(disp_y))
  #           # print(disp_x)
  #           # print(disp_y)
  #           bb = ll['bbox']
  #           # center = (int(bb[0]), int(bb[1]))
  #           center = (int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2))
  #           if disp_x > 0.0095:
  #               f.write(str(center[0])+','+str(center[1])+','+str(disp_x)+','+str(disp_y)+'\n')
  #               cv2.circle(img2, (center[0], center[1]), 5, (255,0,0), -1)
  #           # cv2.imshow("Image", img2)
  #           # cv2.waitKey(0)
  #         f.close()
  # print("Mean_x : "+str(sum(all_x_disp)/len(all_x_disp)))
  # print("Mean_y : "+str(sum(all_y_disp)/len(all_y_disp)))
  # print("Minimum value of x: "+str(min(all_x_disp)))
  # print("Maximum value of x: "+str(max(all_x_disp)))
  # cv2.destroyAllWindows()
              
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
