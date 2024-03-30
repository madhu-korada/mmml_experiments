import os
import torch 
import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor

from analysis.analyze_clip import get_image_info
from llava_evaluate import load_gt, load_llaava_output

model_card = "openai/clip-vit-large-patch14-336"
clip_model = CLIPModel.from_pretrained(model_card)
clip_processor = CLIPProcessor.from_pretrained(model_card)
tokenizer = AutoTokenizer.from_pretrained(model_card)
cosi = torch.nn.CosineSimilarity(dim=0) 

model = model_card.split("/")[-1]

def get_cosine_similarity(image, question_string, answer_string):
    inputs = tokenizer([question_string + " " + answer_string], padding=True, return_tensors="pt")
    text_features = clip_model.get_text_features(**inputs).squeeze(dim=0)
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs).squeeze(dim=0)
    output = cosi(image_features, text_features) 
    return output.item()


if __name__ == "__main__":
    ok_vqa_gt_file = 'data/gt_ok_vqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    ann_file = 'data/gt_ok_vqa/instances_val2014.json'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'
    clip_predictions_file = f'data/clip_predictions-{model}.jsonl'

    llava_answers = load_llaava_output(llava_output_single_word_file)
    gold_answers, image_ids = load_gt(ok_vqa_gt_file)
    
    cosine_similarities_file = 'data/eval_llava/clip_cosine_similarities.jsonl'
    
    # if not DO_MODEL_INFERENCE:
    #     clip_predictions = load_llaava_output(clip_predictions_file)
    ctr = 0
    # if cosine_similarities_file exists open it and read the lines
    if os.path.exists(cosine_similarities_file):
        with open(cosine_similarities_file, 'r') as f:
            lines = f.readlines()
        ctr = len(lines)
        # filter the image_ids that have already been processed
        # image_ids = {k: v for k, v in image_ids.items() if v not in [int(line['image_id']) for line in lines]}
        # llava_answers = [line for line in llava_answers if line['question_id'] not in [int(line['question_id']) for line in lines]]
        
    for line in llava_answers:
        ctr += 1
        image_id = image_ids[line['question_id']]
        image_file = f'data/coco2014/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg'
        
        image = Image.open(image_file)
        question_id = line['question_id']
        question = line['prompt']
        answer = line['text']
        cosine_similarity = get_cosine_similarity(image, question, answer)

        json_str = f'{{"question_id": {question_id}, "image_id": {image_id}, "question": "{question}", "answer": "{answer}", "cosine_similarity": {cosine_similarity}}}'
        with open(cosine_similarities_file, 'a') as f:
            f.write(json_str + '\n')
        print(f'-----------------------------------')
        print(f'{ctr}: {json_str}')
      