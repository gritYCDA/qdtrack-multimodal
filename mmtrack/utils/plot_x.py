import torch
import mmcv
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

import os
import os.path as osp

mean = torch.Tensor([123.675, 116.28, 103.53])
std = torch.Tensor([58.395, 57.12, 57.375])

# caffe version
mean = torch.Tensor([103.530, 116.280, 123.675])
std = torch.Tensor([1.0, 1.0, 1.0])

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def plot_img(img):
  n, c, h, w = img.shape
  
  img = (img[0].cpu() * std.view(-1, 1, 1)) + mean.view(-1,1,1)
  img = img.permute(1,2,0).cpu().numpy().astype(np.int32)

  plt.imshow(img)
  plt.show()

def plot_det_bboxes(img,
                    det_bboxes,
                    bbox_color = 'green',
                    text_color = 'green',
                    thickness = 2,
                    font_size = 13,
                    out_directory='exps',
                    file_name='test'):
  EPS = 1e-2 
  n, c, h, w = img.shape

  bbox_color = color_val_matplotlib(bbox_color)
  text_color = color_val_matplotlib(text_color)

  img = (img[0].cpu() * std.view(-1, 1, 1)) + mean.view(-1,1,1)
  img = img.permute(1,2,0).cpu().numpy().astype(np.int32)

  fig = plt.figure()
  canvas = fig.canvas
  dpi = fig.get_dpi()
  fig.set_size_inches((w + EPS) / dpi, (h + EPS) / dpi)

  plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
  ax = plt.gca()
  ax.axis('off')
  
  polygons = []
  color = []
  for i, bbox in enumerate(det_bboxes):
    bbox_int = bbox.cpu().numpy().astype(np.int32)              
    poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
            [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
    np_poly = np.array(poly).reshape((4, 2))
    polygons.append(Polygon(np_poly))
    color.append(bbox_color)
    if len(bbox) > 4:
      label_text = f'{bbox[-1]:.02f}'
    else:
      label_text = str(i)
    ax.text(
      bbox_int[0],
      bbox_int[1],
      f'{label_text}',
      bbox={
        'facecolor': 'black',
        'alpha': 0.8,
        'pad': 0.7,
        'edgecolor': 'none'
      },
      color=text_color,
      fontsize=font_size,
      verticalalignment='top',
      horizontalalignment='left')

  print(len(polygons))
  plt.imshow(img)
  p = PatchCollection(
    polygons, facecolor='none', edgecolors=color, linewidths=thickness)
  ax.add_collection(p)

  plt.show()
  if not osp.exists(out_directory):
    os.makedirs(out_directory)
  plt.savefig(osp.join(out_directory, file_name))

  plt.close()


def plot_tracklets(img,
                    bboxes,
                    labels=None,
                    track_ids=None,
                    id2color=None,
                    bbox_color = 'green',
                    text_color = 'green',
                    thickness = 1,
                    font_size = 5,
                    dir_name='exps',
                    file_name='test',
                    valid_ids=None):
  
  c_path = '/data1/tao/annotations/tao_classes.txt'
  categories = open(c_path, 'r').readlines()
  categories = [cat[:-1] for cat in categories]
  
  EPS = 1e-2 
  n, c, h, w = img.shape

  if track_ids is None:
    bbox_color = color_val_matplotlib(bbox_color)
  text_color = color_val_matplotlib(text_color)

  img = (img[0].cpu() * std.view(-1, 1, 1)) + mean.view(-1,1,1)
  img = img.permute(1,2,0).cpu().numpy().astype(np.int32)

  fig = plt.figure()
  canvas = fig.canvas
  dpi = fig.get_dpi()
  fig.set_size_inches((w + EPS) / dpi, (h + EPS) / dpi)

  plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
  ax = plt.gca()
  ax.axis('off')
  
  polygons = []
  colors = []
  if track_ids is not None and id2color is None:
    id2color = {}

  for i, bbox in enumerate(bboxes):
    bbox_int = bbox.cpu().numpy().astype(np.int32)              
    poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
            [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
    np_poly = np.array(poly).reshape((4, 2))
    polygons.append(Polygon(np_poly))
    if track_ids is not None:
      if track_ids[i].item() not in id2color:
        bbox_color = color_val_matplotlib(np.random.randint(255, size=3))
        id2color[track_ids[i].item()] = bbox_color
      else:
        bbox_color = id2color[track_ids[i].item()]
    
    if valid_ids is not None:
      if valid_ids[i] == True:
        bbox_color = 'green'
      else:
        bbox_color = 'red'
    colors.append(bbox_color)

    c = None
    if labels is not None:
      c = categories[labels[i]]
    if len(bbox) > 4:
      label_text = f'{c}, {bbox[-1]:.02f}'
    else:
      label_text = c

    ax.text(
      bbox_int[0],
      bbox_int[1],
      f'{label_text}',
      bbox={
        'facecolor': 'black',
        'alpha': 0.8,
        'pad': 0.7,
        'edgecolor': 'none'
      },
      color=text_color,
      fontsize=font_size,
      verticalalignment='top',
      horizontalalignment='left')

    if track_ids is not None:
      id_text = f'{track_ids[i]}'
      ax.text(
        bbox_int[2],
        bbox_int[1],
        f'{id_text}',
        bbox={
          'facecolor': 'black',
          'alpha': 0.8,
          'pad': 0.7,
          'edgecolor': 'none'
        },
        color=text_color,
        fontsize=font_size,
        verticalalignment='top',
        horizontalalignment='left')

  # print(len(polygons))
  plt.imshow(img)
  p = PatchCollection(
    polygons, facecolor='none', edgecolors=colors, linewidths=thickness)
  ax.add_collection(p)

  if not osp.exists(dir_name):
    os.makedirs(dir_name)
  plt.savefig(osp.join(dir_name, file_name))
  plt.close()
  return id2color