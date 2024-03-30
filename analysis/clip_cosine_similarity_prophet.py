import os
import json
import torch 
import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor

from analysis.analyze_clip import get_image_info
from llava_evaluate import load_gt, load_llaava_output
from prophet_evaluate import load_prophet_output, load_questions

model_card = "openai/clip-vit-large-patch14-336"
clip_model = CLIPModel.from_pretrained(model_card, device_map="cuda")
clip_processor = CLIPProcessor.from_pretrained(model_card, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_card)
cosi = torch.nn.CosineSimilarity(dim=0) 

model = model_card.split("/")[-1]

def get_cosine_similarity(image, question_string, answer_string):
    # Tokenize the text input
    text_inputs = tokenizer([question_string + " " + answer_string], padding=True, return_tensors="pt", truncation=True, max_length=77)
    text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
    # Get text features
    text_features = clip_model.get_text_features(**text_inputs).squeeze(dim=0)
    
    # Process the image input and move to CUDA
    image_inputs = clip_processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to("cuda") for k, v in image_inputs.items()}
    # Get image features
    image_features = clip_model.get_image_features(**image_inputs).squeeze(dim=0)
    
    # Calculate cosine similarity
    output = cosi(image_features, text_features)
    return output.item()

   
def get_qid_given_question(question, prophet_questions):
    for k, v in prophet_questions.items():
        if v == question:
            return k
    return None
    
def get_line_given_qid(question_id, prophet_answers):
    for line in prophet_answers:
        if line['question_id'] == question_id:
            return line
    return None

if __name__ == "__main__":
    EVAL_MODE = 'prophet'
    ok_vqa_gt_file = 'data/gt_ok_vqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    ann_file = 'data/gt_ok_vqa/instances_val2014.json'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'
    prophet_output_file = 'data/eval_prophet/result_20240228154916.json'
    clip_predictions_file = f'data/clip_predictions-{model}.jsonl'

    gold_answers, image_ids = load_gt(ok_vqa_gt_file)
    llava_answers = load_llaava_output(llava_output_single_word_file)
    prophet_answers = load_prophet_output(prophet_output_file)
    prophet_questions = load_questions()
    
    cosine_similarities_file = f'data/eval_prophet/clip_cosine_similarities_{EVAL_MODE}.jsonl'
    
    ctr = 0
    # if cosine_similarities_file exists open it and read the lines
    if os.path.exists(cosine_similarities_file):
        with open(cosine_similarities_file, 'r') as f:
            lines = f.readlines()
        ctr = len(lines)
          
        processed_questions_ids = {json.loads(line)['question_id'] for line in lines}
        # filter the llava answers that have already been processed
        prophet_answers = [line for line in prophet_answers if line['question_id'] not in processed_questions_ids]
        
    print(f'Number of questions to process: {len(prophet_answers)}')
    for line in prophet_answers:
        ctr += 1
        question_id = line['question_id']
        question = prophet_questions[question_id]
        answer = line['answer']
        image_id = image_ids[question_id]
        image_file = f'data/coco2014/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg'
        
        image = Image.open(image_file)
        
        cosine_similarity = get_cosine_similarity(image, question, answer)

        json_str = f'{{"question_id": {question_id}, "image_id": {image_id}, "question": "{question}", "answer": "{answer}", "cosine_similarity": {cosine_similarity}}}'
        with open(cosine_similarities_file, 'a') as f:
            f.write(json_str + '\n')
        print(f'-----------------------------------')
        print(f'{ctr}: {json_str}')
    
    if ctr == len(prophet_answers) or len(prophet_answers) == 0:
        print(f'All questions have been processed. Calculating avg cosine similarity...')
        with open(cosine_similarities_file, 'r') as f:
            lines = f.readlines()
        total = 0
        for line in lines:
            total += json.loads(line)['cosine_similarity']
        avg = total / len(lines)
        print(f'Average cosine similarity: {avg}')