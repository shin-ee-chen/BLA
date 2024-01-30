import numpy as np
import copy
import torch
from transformers import logging
# logging.set_verbosity_error()


import os
import json
import random

import wandb

import argparse
import numpy as np
from tqdm import tqdm
import time

from magma import Magma
from magma.image_input import ImageInput

import utils

TABLE_NAME = ""


def prompt_question(caption):
    template = "Q: {} A:{}"
    question = "Is the sentence {} appropriate for this image? Only reply yes or no.".format(caption)

    return template.format(question, "")


def in_concext_learning_prompt(true_exp, false_exp, caption):
    template = "Q: {} A: {}"
    question = "Is the sentence {} appropriate for this image? Only reply yes or no."
    
    # Randomly shuffle the true and false example order
    if random.random() > 0.5:
        context = [
            (question.format(true_exp), "yes."),
            (question.format(false_exp), "no."),
            (question.format(caption), "")
        ]
        
    else:
        context = [
            (question.format(false_exp), "no."),
            (question.format(true_exp), "yes."),
            (question.format(caption), "")
        ]
        
    return " ".join([template.format(context[i][0], context[i][1]) \
           for i in range(len(context))])
  

def test_magma(args):
    # Task name
    if "active" in args.file_path.lower():
        task_name = "ctrl_ap_tasks"
    elif "coord" in args.file_path.lower():
        task_name = "ctrl_coord_tasks"
    else:
        task_name = "ctrl_rc_tasks"
    
    dataset_name = args.file_path.split("/")[-2] if args.dataset_type == "test" else \
        args.file_path.split("/")[-1].split(".")[0]
    
    if args.in_context_learning:
        prompt_type = "in_context_cross" if args.cross_dataset_example else "in_context_same"
    else:
        prompt_type = "prompt"
        
    model_name = "magma_" + prompt_type
    
    if args.use_wandb:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="bla-2023",
            # track hyperparameters and run metadata
            config={
                    "dataset": dataset_name,
                    "model_name": model_name
                    }
            )     
    
    items = utils.read_json_file(args.file_path)
    if args.dataset_type == "test":
        items = utils.preprocess_json_format(items)
    
    
    
    if args.cross_dataset_example:
        example_dataset = utils.read_json_file_as_dict(args.example_file_path)


    model = Magma.from_checkpoint(
        config_path = "magma/configs/MAGMA_v1.yml",
        checkpoint_path = "models/magma_model/mp_rank_00_model_states.pt",
        device = 'cuda:0'
    )
    # # 
    # ret_scale_factor = 1.0 # Increase this hyperparameter to upweight the probability of magma returning an image.
    caption_types = ['True1', 'True2', 'False1', 'False2']
    
    # acc = 0
    answers = []
    predictions = {}
    
    TP, FP, TN, FN = 0, 0, 0, 0
    count = 0
    for item in tqdm(items):
        # if count >= 5:
        #     break
        count += 1
        image_id = item['image_id']
        captions = item['caption_group'][0]
        
        # Skip test data that doesn't exist in the example dataset
        if args.cross_dataset_example and image_id not in example_dataset:
            continue
        
        img_path =  "BLA/images/" + str(image_id) + ".jpg"
        inp_image = ImageInput(img_path)
        
        predictions[image_id] = []
        
        total_sentence = 0
        error_count = 0
        
        for i, type in enumerate(caption_types):
            total_sentence += 1
            if args.in_context_learning:
                if args.cross_dataset_example:
                    exp_captions = example_dataset[image_id]
                    
                    prompt = in_concext_learning_prompt(exp_captions["True1"], 
                                                    exp_captions["False1"], 
                                                    captions[type]) if i % 2 != 0\
                            else in_concext_learning_prompt(exp_captions["True2"], 
                                                        exp_captions["False2"], 
                                                        captions[type])
                    
                else:
                    prompt = in_concext_learning_prompt(captions["True1"], 
                                                    captions["False1"], 
                                                    captions[type]) if i % 2 != 0\
                            else in_concext_learning_prompt(captions["True2"], 
                                                        captions["False2"], 
                                                        captions[type])
              
            else:
                prompt = prompt_question(captions[type])
            
            text = prompt.strip()
            inputs = [
                inp_image,
                text
            ]
            
            embeddings = model.preprocess_inputs(inputs) 
            out = model.generate(
                embeddings = embeddings,
                max_steps = 6,
                temperature = 0.7,
                top_k = 0,
                )  
            
            result = out[0].strip().lower()
            
            if len(result) != 0 and result[-1] == ".":
                result = result[:-1]
            
            if "appropriate" in result:
                   result = "yes"
            
            
         
            answers.append(result)
            predictions[image_id].append(result)
            
            
            if "yes" in result:
                TP += 1 if i < 2 else 0
                FP += 1 if i >= 2 else 0
            
            elif "no" in result:
                TN += 1 if i >= 2 else 0
                FN +=1 if i < 2 else 0
            
            else:
                print("Error: ", result, image_id, type)
                error_count += 1
           
    # Performance statistics
    bi_cls_results = utils.get_cls_statistics(TP, FP, TN, FN)
    
    print("Performance:", bi_cls_results)
    print(f"error_count = {error_count}, total_sentence={total_sentence}, percentage={error_count/total_sentence}")
    
    if args.use_wandb:
        # Logging the results
        bi_result_table = wandb.Table(columns=["model", "dataset"] + list(bi_cls_results.keys()), 
                                 data=[[model_name, task_name] + list(bi_cls_results.values())]
                                    )
    
        table_name = "test_set" if args.dataset_type == "test" else "whole_dataset"

        run.log({table_name: bi_result_table})
    
    
    logdir = "magma/scripts"
    file_name = args.file_path.split("/")[-1].split(".")[0] + "_" + prompt_type +"_predictions.json"
    json.dump(predictions, open(os.path.join(logdir, file_name), "w"), indent=2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_path',type = str,\
        default='datasets/BLA/annotations/active_passive_captions.json', \
        help = "path of caption input")
    parser.add_argument('--dataset_type', type = str, default='whole', choices=['test', 'whole'],
                        help="whether the dataset is a test set or the whole evaluation set"
                        )
    parser.add_argument('--in_context_learning', action="store_true" ,\
        help = "whether to use in context learning")
    
    parser.add_argument('--cross_dataset_example', action="store_true" ,\
        help = "whether to use in context learning")
    parser.add_argument('--example_file_path',type = str,\
        default='datasets/BLA/annotations/relative_clause_captions.json', \
        help = "path of caption input")
    
    parser.add_argument('--use_wandb', action="store_true",
                        help = "Whether to use wandb to sore results")
    
    args = parser.parse_args()
    
    test_magma(args)
    
            