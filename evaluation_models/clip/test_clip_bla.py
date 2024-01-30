import json
import os
import clip
import torch
import clip.utils as utils
import argparse

TASK_NAME_TO_FILE = {
    "ap": "active_passive_captions.json",
    "co": "coordination_captions.json",
    "rc": "relative_clause_captions.json"
}


def eval_on_clip(args):
    file_path = os.path.join(args.annotation_dir, TASK_NAME_TO_FILE(args.phenomenon))
    items = utils.read_json_file(file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    clip_results = []

    for item in items:
        image_id = item['image_id']
        captions = item['caption_group'][0]
        img_path = os.path.join(args.img_dir, image_id + ".jpg")
        image = utils.show_image(img_path)

        rank, clip_score = utils.rank_captions(image, 
                                               [captions['True1'], captions['True2'], captions['False1'], captions['False2']], 
                                               clip_model, clip_preprocess, device)
        if "predicate" in captions:
            clip_results.append({"image_id": image_id, "rank": rank, 
                                 "predicate": captions["predicate"]})
        else:
            clip_results.append({"image_id": image_id, "rank": rank})
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    output_results_file = os.path.join(args.output_dir, 
                                       "{}_clip_results.json".format(args.phenomenon))
    utils.write_file(output_results_file, clip_results)
    print("finish\n")

if __name__ == "__main__":
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
    
    parser.add_argument('--output_dir',type = str,
                        default="outputs", 
                        help = "Path for saving model prediction results")
    

    args = parser.parse_args()

    eval_on_clip(args)