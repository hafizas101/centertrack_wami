from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths

import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import pdb
import torch
import torch.utils.data
import sys
# sys.path.append("./lib/model/networks/DCNv2")
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)

  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
#  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  if opt.val_intervals < opt.num_epochs or opt.test:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
      pin_memory=True)

    if opt.test:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir)
      return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
  )
  tb = SummaryWriter()

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    #pdb.set_trace()
    log_dict_train, _ = trainer.train(epoch, train_loader)
    tb.add_scalar('Loss', log_dict_train['tot'], epoch)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      print(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)))
      print("IN OPT VAL INTERVALS")
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if opt.eval_val:
          val_loader.dataset.run_eval(preds, opt.save_dir)
    else:
      print("IN THE ELSE STATEMENT")
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
#    logger.write('\n')
    if epoch in opt.save_point:
      print("IN 2nd if statement")
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      print("IN 3rd IF statement")
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  tb.close()
#  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
