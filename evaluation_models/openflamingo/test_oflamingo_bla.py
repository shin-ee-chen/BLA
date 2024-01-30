from model_container import model_import
import torch
import os
import json
import wandb

import argparse

from tqdm import tqdm

import bla_utils as utils

TASK_NAME_TO_FILE = {
    "ap": "active_passive_captions.json",
    "co": "coordination_captions.json",
    "rc": "relative_clause_captions.json"
}

def evaluate_vqa(args):
    eval_model = model_import(args.model_name, args)
    
    if args.in_context_learning:
        prompt_type = "in_context_cross" if args.cross_dataset_example else "in_context_same"
    else:
        prompt_type = "prompt"
        
    model_name = f"{args.model_name}_{prompt_type}"
    
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="bla-2023",
    
        # track hyperparameters and run metadata
        config={
        "dataset": args.phenomenon,
        "model_name": model_name
        }
    )     
    
    file_path = os.path.join(args.annotation_dir, TASK_NAME_TO_FILE(args.phenomenon))
    items = utils.read_json_file(file_path)
    if args.dataset_type == "test":
        items = utils.preprocess_json_format(items)
    
    
    if args.cross_dataset_example:
        example_file_path = os.path.join(args.annotation_dir, TASK_NAME_TO_FILE(args.example_task))
        example_dataset = utils.read_json_file_as_dict(example_file_path)

    
    caption_types = ['True1', 'True2', 'False1', 'False2']
    
    answers = []
    predictions = {}
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for item in tqdm(items):
        image_id = item['image_id']
        captions = item['caption_group'][0]
        
        # Skip test data that doesn't exist in the example dataset
        if args.cross_dataset_example and image_id not in example_dataset:
            continue
        
        img_path = os.path.join(args.img_dir, image_id + ".jpg")
        
        predictions[image_id] = []
        
        for i, type in enumerate(caption_types):
            
            if args.in_context_learning:
                if args.cross_dataset_example:
                    exp_captions = example_dataset[image_id]
                    
                    prompt = eval_model.in_concext_learning_prompt(exp_captions["True1"], 
                                                    exp_captions["False1"], 
                                                    captions[type]) if i % 2 == 0\
                            else eval_model.in_concext_learning_prompt(exp_captions["True2"], 
                                                        exp_captions["False2"], 
                                                        captions[type])
                    
                else:
                    prompt = eval_model.in_context_learning_prompt(captions["True1"], 
                                                    captions["False1"], 
                                                    captions[type]) if i % 2 == 0\
                            else eval_model.in_context_learning_prompt(captions["True2"], 
                                                        captions["False2"], 
                                                        captions[type])
                    vision_x = eval_model.image_encoder([img_path])
            else:
                prompt = eval_model.prompt_question(captions[type])
                vision_x = eval_model.image_encoder([img_path])
            
            result = eval_model.get_outputs(prompt, vision_x)
            
            
            if result[-1] == ".":
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
                
           
    # Performance statistics
    bi_cls_results = utils.get_cls_statistics(TP, FP, TN, FN)
    
    print("Performance:", bi_cls_results)
    print(answers.count("yes"))
    print(answers.count("no"))
    
    
    
    # Logging the results
    bi_result_table = wandb.Table(columns=["model", "dataset"] + list(bi_cls_results.keys()), 
                                 data=[[model_name, args.phenomenon] + list(bi_cls_results.values())]
                                 )
    
    table_name = f"{model_name}_results" if args.dataset_type == "test" else "evaluation_results"

    run.log({table_name: bi_result_table})
    
    
    logdir = "outputs"
    file_name = args.phenomenon + "_" + prompt_type +"_predictions.json"
    json.dump(predictions, open(os.path.join(logdir, file_name), "w"), indent=2)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--annotation_dir',type = str,
                        default='bla_benchmark/annotations', 
                        help = "Path of benchmark annotations")
    
    parser.add_argument('--img_dir',type = str,
                        default='bla_benchmark/images', 
                        help = "Path of benchmark images")
    
    parser.add_argument('--phenomenon',type = str, 
                        default="ap",
                        choices=["ap", "co", "rc"], 
                        help = "Phenomenon task choice: ac, co, or rc")
    
    parser.add_argument('--device', type = str, default=device, 
                        help="cuda or cpu"
                        )
    parser.add_argument('--dataset_type', type = str, default='test', choices=['test', 'whole'],
                        help="whether the dataset is a test set or the whole evaluation set"
                        )
    parser.add_argument('--model_name', type = str, default='FLAMINGO', choices=['FLAMINGO', 'BLIP'],
                        help="Candidate evaluation model"
                        )
    parser.add_argument('--in_context_learning', action="store_true" ,\
        help = "whether to use in context learning")
    
    parser.add_argument('--cross_dataset_example', action="store_true" ,\
        help = "whether to use in context learning")
    
    args = parser.parse_args()
    
    evaluate_vqa(args)
    
            