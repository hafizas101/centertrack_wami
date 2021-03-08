from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ..generic_dataset import GenericDataset

class WPAFB(GenericDataset):
  num_categories = 1
  default_resolution = [-1, -1]
  #class_name = ['ignore', 'pedestrain','people','bicycle','car','van','truck','tricycle','awning-tricycle', 'bus',  'motor','others']
  class_name = ['vehicle']
  max_objs = 128
  cat_ids = {1: 1, -1: -1}
  cat_id_2_class_id ={1:1, 0:0, -1:-1}
  def __init__(self, opt, split):
    assert (opt.custom_dataset_img_path != '') and \
      (opt.custom_dataset_ann_path != '') and \
      (opt.num_classes != -1) and \
      (opt.input_h != -1) and (opt.input_w != -1), \
      'The following arguments must be specified for custom datasets: ' + \
      'custom_dataset_img_path, custom_dataset_ann_path, num_classes, ' + \
      'input_h, input_w.'
    img_dir = opt.custom_dataset_img_path
    ann_path = opt.custom_dataset_ann_path
    self.num_categories = opt.num_classes
    #self.class_name = ['' for _ in range(self.num_categories)]
    self.default_resolution = [opt.input_h, opt.input_w]
    #self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

    self.images = None
    # load image list and coco
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded Custom dataset {} samples'.format(self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))


  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_wpafb_r2_speed_mov_tracking')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)

    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      
      for image_info in images:
        img_id = image_info['id']
        if not (img_id in results):
          continue
        frame_id = image_info['frame_id'] 
        for i in range(len(results[img_id])):
          item = results[img_id][i]
          if not ('alpha' in item):
            item['alpha'] = -1
          if not ('rot_y' in item):
            item['rot_y'] = -10
          if 'dim' in item:
            item['dim'] = [max(item['dim'][0], 0.01), 
              max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
          if not ('dim' in item):
            item['dim'] = [-1, -1, -1]
          if not ('loc' in item):
            item['loc'] = [-1000, -1000, -1000]
          
          track_id = item['tracking_id'] if 'tracking_id' in item else -1
          #HFA -DİKKAT!
          f.write('{} {} '.format( int(images[frame_id-1]['file_name'][-8:-4]), track_id ))
          f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            item['bbox'][0], item['bbox'][1], item['bbox'][2]-item['bbox'][0]+1, item['bbox'][3]-item['bbox'][1]+1))
          
          f.write(' {:.2f} {:d} '.format(item['score'], self.cat_id_2_class_id[item['class']]))
          f.write('-1 -1 \n')
      f.close()

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    #os.system('python tools/eval_kitti_track/evaluate_tracking.py ' + \
    #          '{}/results_visdrone_tracking/ {}'.format(
    #            save_dir, self.opt.dataset_version))
