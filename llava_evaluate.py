import os
import re
import json
import nltk
import torch
import spacy
from torch.cuda import get_device_properties
from fuzzywuzzy import fuzz
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Example input text
input_prompt = "Please evaluate my prediction compared to the ground truth. Each question has multiple ground truth answers in a list (some of them might be same). The 10 ground truth answers are in a list. The prediction is evaluated against each ground truth answer. The final score is the average of the scores for each ground truth answer. The score for each ground truth answer is 1 if the prediction matches the ground truth answer exactly, and 0 otherwise. The final score is between 0 and 10. The higher the score, the better the prediction. The score is 10 if the prediction matches all the ground truth answers exactly, and 0 otherwise. The score is 5 if the prediction matches half of the ground truth answers exactly. Since there are 10 ground truth answers, each correct answer will correspond to 1. Now return the value as a single integer. i.e. output a single int. "

use_model_flag = False

def check_gpu_vram_less_than_8GB():
    if not torch.cuda.is_available():
        return False
    gpu_props = get_device_properties(0) # Assuming single GPU setup
    total_memory_GB = gpu_props.total_memory / (1024**3) # Convert bytes to GB
    print(f"Total GPU memory: {total_memory_GB:.2f} GB")
    return total_memory_GB < 8

if check_gpu_vram_less_than_8GB():
    print("GPU VRAM less than 8GB. Using 8-bit quantization.")
    os.environ['LD_LIBRARY_PATH'] = ''
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None


llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
llava_output_single_word_file = 'data/eval_llava/answer-file-our-single-word.jsonl'
ok_vqa_gt_file = 'prophet/datasets/okvqa/mscoco_val2014_annotations.json'

def load_llaava_output(file_path):
    with open(file_path, 'r') as f:
        llava_output = f.readlines()
        llava_output = [eval(x) for x in llava_output]
    return llava_output

def load_json(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data

def load_gt(file_path):
    gold_answers = {}
    gt_data = load_json(file_path)
    for annotation in gt_data['annotations']:
        img_id = annotation['image_id']
        question_id = annotation['question_id']
        answers = annotation['answers']
        gt_answer = [answer_info['answer'] for answer_info in answers]
        img_key = 'COCO_val2014_{}.jpg#{}'.format(str(img_id).zfill(12), question_id)
        # gold_answers[img_key] = gt_answer
        gold_answers[question_id] = gt_answer
    return gold_answers

# Normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Normalize and tokenize
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'\W', ' ', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # return tokens
    return set(tokens)

def get_num_matches(gold_answers, llava_answer):
    gold_answers_count = Counter(gold_answers)
    
    gold_keywords = preprocess(' '.join(gold_answers))
    llava_keywords = preprocess(llava_answer)
    
    matches = gold_keywords.intersection(llava_keywords)
    match_count = len(matches)
    gold_count = {k: gold_answers_count[k] for k in matches}
    num_gold = sum(gold_count.values())
    return matches, match_count, num_gold


def get_num_matches_spacy(gold_answers, llava_answer):
    gold_answers_count = Counter(gold_answers)
    # Prepare spaCy Docs
    llava_doc = nlp(normalize_text(llava_answer))
    gold_docs = [nlp(normalize_text(answer)) for answer in set(gold_answers)]  # Using set to remove duplicates

    # Matching using Semantic Similarity and Fuzzy Matching
    matches = []
    for gold_doc in gold_docs:
        similarity = llava_doc.similarity(gold_doc)
        # Threshold for similarity can be adjusted
        if similarity > 0.8:
            matches.append(gold_doc.text)
        else:
            # Fallback to fuzzy matching for close misses
            if fuzz.partial_ratio(normalize_text(llava_answer), gold_doc.text) > 80:
                matches.append(gold_doc.text)

    match_count = len(matches)
    gold_count = {k: gold_answers_count[k] for k in matches}
    num_gold = sum(gold_count.values())
    return matches, match_count, num_gold









if not use_model_flag:
    cnt = 1
    # Load gold_answers and predctions
    gold_answers = load_gt(ok_vqa_gt_file)
    llava_answers = load_llaava_output(llava_output_file)
    
    total_match_count = 0
    total_gold_count = 0
    
    atleast_one_match = 0
    
    for line in llava_answers:
        question_id = line['question_id']
        question = line['prompt'].split("\n")[0]
        llava_answer = line['text']
        gold_answer = gold_answers[question_id]
        
        # matches, match_count, num_gold = get_num_matches(gold_answer, llava_answer)
        matches, match_count, num_gold = get_num_matches_spacy(gold_answer, llava_answer)
        # if num_gold < 5:
        print(f'Question: {question} \nGold: {gold_answer} \nLLAVA: {llava_answer}')
        
        print(f'Matches: {matches} Match count: {match_count} Gold count: {num_gold}/{len(gold_answer)}')
        print("\n")
        
        total_match_count += num_gold
        total_gold_count += len(gold_answer)
        if num_gold > 0:
            atleast_one_match += 1
        
        cnt += 1
        if cnt > 10:
            break
    
    print(f'Total match count: {total_match_count}, Total gold count: {total_gold_count}, Atleast one match: {atleast_one_match}')
    print(f'Accuracy: {total_match_count/total_gold_count*100:.2f}%')
    print(f'Atleast one match accuracy: {atleast_one_match/cnt*100:.2f}%')


























    # model_id = "gg-hf/gemma-2b-it"
    # # model_id = "meta-llama/Llama-2-7b-chat-hf"
    # # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # # model_id = "mistralai/Mistral-7B-v0.1"
    # # model_id = "gpt2"
    # dtype = torch.bfloat16

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="cuda",
    #     torch_dtype=dtype,
    #     use_auth_token=True,
    #     quantization_config=quantization_config
    # )
    # gold_answers = load_gt(ok_vqa_gt_file)
    # cnt = 0
    # for line in llava_output_single_word:
    #     question_id = line['question_id']
    #     question = line['prompt']
    #     answer = line['text']
    #     if question_id in gold_answers:
    #         print(question, gold_answers[question_id], answer)
    #         print("\n")
    #     question = question.replace("Answer the question in a single word. i.e. output a single word.", "")
    #     input_text = input_prompt + "\nQuestion: " + question
    #     input_text = input_text + "\nAnswer: " + answer
    #     input_text = input_text + "\nGround Truth: " + str(gold_answers[question_id])
    #     input_text = input_text + "\nScore: "

    #     input_ids = tokenizer.encode(input_text, return_tensors="pt")


    #     # Generate output
    #     # Note: setting temperature=0 would not make sense with beam search as temperature affects sampling, not beam search.
    #     # output_ids = model.generate(input_ids, max_length=50, num_beams=num_beams, temperature=temperature, early_stopping=True)
    #     output_ids = model.generate(input_ids=input_ids.to(model.device), max_new_tokens=10)
    #     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     print(output_text)

    # # beam search for decoding, where sampling temperature is set to 0.





