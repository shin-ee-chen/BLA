from enum import Enum
import subprocess
import sys
import shutil
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from io import BytesIO

import random


from torchvision import transforms
from PIL import Image
# import requests
import json
import numpy as np
import random

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
  """Logs git status to stdout."""
  subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
  subprocess.call('echo', shell=True, stdout=out_file)
  exclude_string = ''
  subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def truncate_caption(caption: str) -> str:
  """Truncate captions at periods and newlines."""
  caption = caption.strip('\n')
  trunc_index = caption.find('\n') + 1
  if trunc_index <= 0:
      trunc_index = caption.find('.') + 1
  if trunc_index > 0:
    caption = caption[:trunc_index]
  return caption


def pad_to_size(x, size=256):
  delta_w = size - x.size[0]
  delta_h = size - x.size[1]
  padding = (
    delta_w // 2,
    delta_h // 2,
    delta_w - (delta_w // 2),
    delta_h - (delta_h // 2),
  )
  new_im = ImageOps.expand(x, padding)
  return new_im


class RandCropResize(object):

  """
  Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
  """

  def __init__(self, target_size):
    self.target_size = target_size

  def __call__(self, img):
    img = pad_to_size(img, self.target_size)
    d_min = min(img.size)
    img = T.RandomCrop(size=d_min)(img)
    t_min = min(d_min, round(9 / 8 * self.target_size))
    t_max = min(d_min, round(12 / 8 * self.target_size))
    t = random.randint(t_min, t_max + 1)
    img = T.Resize(t)(img)
    if min(img.size) < 256:
      img = T.Resize(256)(img)
    return T.RandomCrop(size=self.target_size)(img)


class SquarePad(object):
  """Pads image to square.
  From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
  """
  def __call__(self, image):
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return F.pad(image, padding, 0, 'constant')


def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img


def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
  print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
  feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
  return feature_extractor


def get_pixel_values_for_model(feature_extractor, img):
  pixel_values = feature_extractor(
    img.convert('RGB'),
    return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
  return pixel_values


def save_checkpoint(state, is_best, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if is_best:
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def accuracy(output, target, padding, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res


def get_params_count(model, max_name_len: int = 60):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)


#######################BLA_PROJECT#########################
def load_demo_image(image_size,device, img_path):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    raw_image = Image.open(img_path).convert('RGB')   
    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def read_json_file(file_path):
    global json_file
    try:
        json_file = open(file_path, "r")
        return json.load(json_file)
    finally:
        if json_file:
            print("close file...")
            json_file.close()


def read_json_file_as_dict(file_path):
    json_file = preprocess_json_format(read_json_file(file_path))
    json_dict = {}
    for item in json_file:
        image_id = item['image_id']
        captions = item['caption_group'][0]
        json_dict[image_id] = captions
        
    return json_dict


def preprocess_json_format(json_file):
    # unified the test json file to the whole dataset one
    if isinstance(json_file, list):
        return json_file
    
    caption_types = ['True1', 'True2', 'False1', 'False2']
    new_json_file = []
    for img_id, captions in json_file.items():
        caption_group = {caption_types[i]: caption for i, caption in enumerate(captions)}
        new_json_file.append({"image_id": img_id, "caption_group": [caption_group]})
    
    return new_json_file


def get_rank_count(ranks):
    rank_count = {}
    
    if not isinstance(ranks, list):
        ranks = ranks.tolist()
    
    for rank in ranks:
        str_rank = [str(r) for r in rank]
        rank_key = ''.join(str_rank)
        if rank_key in rank_count:
            rank_count[rank_key] += 1
        else:
            rank_count[rank_key] = 1
    return rank_count
    
def get_sent_acc(rank):
    correct_count = 0
    total = len(rank)
    for i, r in enumerate(rank):
        r = int(r)
        if i < total/2 and r <= total/2:
            correct_count += 1
        elif i >= total/2 and r > total/2:
            correct_count += 1
    return correct_count/total


def get_true_false_prediction_num(rank):
    TP, FP, TN, FN = 0, 0, 0, 0
    total = len(rank)
    for i, r in enumerate(rank):
        r = int(r)
        # caption in position 1 and 2
        if i < total/2:
            TP += 1 if r <= total/2 else 0
            FN += 1 if r > total/2 else 0
        
        # caption in position 3 and 4
        else:
            TN += 1 if r > total/2 else 0
            FP += 1 if r <= total/2 else 0
        
    return TP, FP, TN, FN
  

def get_rank_statistics(results):
    ranks = get_rank_count(results)
    total = 0
    # correct count
    set_acc = error_rate = ta_fa = tp_fp = tp_fa = ta_fp= 0
    ta_tp = fa_fp = 0
    sent_acc = 0
    
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    for r, c in ranks.items():
        total += c
        sent_acc += get_sent_acc(r) * c
        TP, FP, TN, FN = get_true_false_prediction_num(r)
        
        total_TP += TP * c
        total_FP += FP * c
        total_TN += TN * c
        total_FN += FN * c
        
        # if len(r) < 3:
        #     continue
        
        # if int(r[0]) < 3 and int(r[1]) < 3:
        #     set_acc += c
        # if int(r[2]) < 3 and int(r[3]) < 3:
        #     error_rate += c
        # if int(r[0]) < int(r[2]):
        #     ta_fa += c
        # if int(r[1]) < int(r[3]):
        #     tp_fp += c
        # if int(r[1]) < int(r[2]):
        #     tp_fa += c
        
        # if int(r[0]) < int(r[3]):
        #     ta_fp += c
        
        # if int(r[0] < r[1]):
        #     ta_tp += c
        
        # if int(r[2] < r[3]):
        #     fa_fp += c
    
    y_precision = 0 if (total_TP + total_FP) == 0 else \
        total_TP / (total_TP + total_FP)
        
    y_recall = 0 if (total_TP + total_FN) == 0 else \
        total_TP / (total_TP + total_FN)
        
    y_f1_score = 0 if (y_recall + y_precision) == 0 else \
        2 * (y_recall * y_precision) / (y_recall + y_precision)
    
    n_precision = total_TN / (total_TN + total_FN)
    n_recall = total_TN / (total_FP + total_TN)
    n_f1_score = 0 if (n_recall + n_precision) == 0 else \
        2 * (n_recall * n_precision) / (n_recall + n_precision)
    
    return {
        "rank_acc": np.round(sent_acc / total * 100, 2),
        "y_f1_score": np.round(y_f1_score * 100, 2),
        "y_precision": np.round(y_precision * 100, 2),
        "y_recall": np.round(y_recall * 100, 2),
        "n_f1_score": np.round(n_f1_score * 100, 2),
        "n_precision": np.round(n_precision * 100, 2),
        "n_recall": np.round(n_recall * 100, 2),
         
        # "set_acc": np.round(set_acc / total * 100, 2),
        # "error_rate": np.round(error_rate / total * 100, 2),
        # "ta_fa / TP1_FP1": np.round(ta_fa / total * 100, 2),
        # "tp_fp / TP2_FP2": np.round(tp_fp / total * 100, 2),
        # "tp_fa / TP2_FP1": np.round(tp_fa / total * 100, 2),
        # "ta_fp / TP1_FP2": np.round(ta_fp / total * 100, 2),
        # "ta_tp / TP1_TP2": np.round(ta_tp / total * 100, 2),
        # "fa_fp / FP1_FP2": np.round(fa_fp / total * 100, 2),
        
        "total": total,
    }
    

def get_cls_statistics(total_TP, total_FP, total_TN, total_FN):
    total = total_TP + total_FP + total_TN + total_FN
    y_precision = total_TP / (total_TP + total_FP)
    y_recall = total_TP / (total_TP + total_FN)
    y_f1_score = 0 if (y_recall + y_precision) == 0 else \
        2 * (y_recall * y_precision) / (y_recall + y_precision)
    
    n_precision = total_TN / (total_TN + total_FN)
    n_recall = total_TN / (total_FP + total_TN)
    n_f1_score = 0 if (n_recall + n_precision) == 0 else \
        2 * (n_recall * n_precision) / (n_recall + n_precision)
    
    accuracy = (total_TP + total_TN) / total
    
    return {
        "cls_acc": np.round(accuracy * 100, 2),
        "cls_y_f1_score": np.round(y_f1_score * 100, 2),
        "cls_y_precision": np.round(y_precision * 100, 2),
        "cls_y_recall": np.round(y_recall * 100, 2),
        "cls_n_f1_score": np.round(n_f1_score * 100, 2),
        "cls_n_precision": np.round(n_precision * 100, 2),
        "cls_n_recall": np.round(n_recall * 100, 2),
        "cls_total": total,
        "cls_num_yes": total_TP + total_FP,
        "cls_num_no": total_TN + total_FN
    }