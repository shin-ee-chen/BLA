from torchvision import transforms
from PIL import Image
# import requests
import json
import numpy as np
import random

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
    

if __name__ == "__main__":
    json_path = "/home/xchen/datasets/BLA/original/finetune/finetune_random/active_passive.json"
    json_file = read_json_file(json_path)
    new_file = preprocess_json_format(json_file)
    print(new_file[0])