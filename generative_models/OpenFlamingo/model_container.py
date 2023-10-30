from PIL import Image
import torch
import random
import os


model_list = [
    "BLIP",
    "FLAMINGO"
]

def model_import(model_name, args):
    assert model_name in model_list, f"{model_name} is not supported"
    
    out_model = eval(f"{model_name}_CONTAINER(args)")
    return out_model

class FLAMINGO_CONTAINER:
    def __init__(self, args):
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", 
            "checkpoint.pt",
        )
        self.args = args
        self.device = self.args.device
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left" 
    
    def image_encoder(self, image_list):
        image_list = [
            Image.open(img_path).convert("RGB") 
            for img_path in image_list
        ]
        vision_x = [self.image_processor(img).unsqueeze(0) for img in image_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        return vision_x
    
    def in_context_learning_prompt(self, true_exp, false_exp, caption):
        template = "<image>Question: {} Short answer: {}"
        question = "Is the sentence {} appropriate for this image?"
        
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
            
        text =  "<|endofchunk|>".join(
            [
                template.format(
                    context[i][0], context[i][1]
                )
                +"\n" 
                for i in range(len(context))
            ]
        )
        return text
    
    def prompt_question(self, caption, answer=None):
        return f"<image>Question: Is the sentence {caption} appropriate for this image? yes or no? \
            Short answer:{'<|endofchunk|>' if answer is not None else ''}"

    def get_outputs(self, prompt, vision_x):

        encodes = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )
        input_ids = encodes['input_ids']
        input_ids = input_ids.to(self.device)
        vision_x = vision_x.to(self.device)
        outputs = self.model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=encodes["attention_mask"],
            max_new_tokens=2000,
            num_beams=2,
        )
        outputs = outputs[:, len(input_ids[0]):]
        
        return (
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split(" ")[-1].strip(". ").lower()
            if self.args.in_context_learning else 
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip(". ").lower()
        )