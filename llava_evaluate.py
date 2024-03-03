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

from f1_score import calculate_f1_score, f1

# Load spaCy model
nltk.download('stopwords')
nlp = spacy.load("en_core_web_md")

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


# def get_num_matches_spacy(gold_answers, llava_answer):
#     gold_answers_count = Counter(gold_answers)
#     # Prepare spaCy Docs
#     llava_doc = nlp(normalize_text(llava_answer))
#     gold_docs = [nlp(normalize_text(answer)) for answer in set(gold_answers)]  # Using set to remove duplicates

#     # Matching using Semantic Similarity and Fuzzy Matching
#     matches = []
#     for gold_doc in gold_docs:
#         similarity = llava_doc.similarity(gold_doc)
#         # Threshold for similarity can be adjusted
#         if similarity > 0.8:
#             matches.append(gold_doc.text)
#         else:
#             # Fallback to fuzzy matching for close misses
#             if fuzz.partial_ratio(normalize_text(llava_answer), gold_doc.text) > 80:
#                 matches.append(gold_doc.text)

#     match_count = len(matches)
#     gold_count = {k: gold_answers_count[k] for k in matches}
#     num_gold = sum(gold_count.values())
#     return matches, match_count, num_gold

# Modified function to include lemmatization
def get_num_matches_spacy(gold_answers, llava_answer):
    gold_answers_count = Counter(gold_answers)
    llava_doc = nlp(normalize_text(llava_answer))
    
    # Extract lemmas for the llava answer
    llava_lemmas = {token.lemma_ for token in llava_doc}
    
    matches = []
    match_details = []

    # Process each unique gold answer through spaCy
    for answer in set(gold_answers):
        gold_doc = nlp(normalize_text(answer))
        
        # Extract lemmas for the gold answer
        gold_lemmas = {token.lemma_ for token in gold_doc}
        
        # Use set intersection for lemmatized match; this may already improve matching
        lemma_matches = gold_lemmas.intersection(llava_lemmas)
        if lemma_matches:
            matches.append(answer)
            match_details.extend(lemma_matches)
        else:
            # Calculate similarity and perform fuzzy matching as fallback
            similarity = llava_doc.similarity(gold_doc)
            if similarity > 0.8 or fuzz.partial_ratio(normalize_text(llava_answer), normalize_text(answer)) > 80:
                matches.append(answer)
    
    match_count = len(set(matches))  # Ensure unique matches
    gold_count = sum(gold_answers_count[match] for match in matches)
    return matches, match_count, gold_count, match_details

def do_exact_matching(gold_answers, llava_answer):
    gold_answers_count = Counter(gold_answers)
    llava_doc = normalize_text(llava_answer)
    
    matches = []
    match_details = []

    # Process each unique gold answer through spaCy
    for answer in set(gold_answers):
        gold_doc = normalize_text(answer)
        
        if gold_doc == llava_doc:
            matches.append(answer)
    
    match_count = len(set(matches))  # Ensure unique matches
    gold_count = sum(gold_answers_count[match] for match in matches)
    return matches, match_count, gold_count, match_details


if __name__ == "__main__":
    use_descriptions = False
    exact_matching = False
    debug_wrong_predictions = False
    
    model_id = "gg-hf/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ok_vqa_gt_file = 'prophet/datasets/okvqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'
    # llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-or-phrase-temp-1-beams-5.jsonl'
    # llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5-next-run.jsonl'
    # llava_output_single_word_file = 'data/eval_llava/answer-file-our-single-word.jsonl'

    cnt = 1
    total_match_count, total_gold_count, atleast_one_match = 0, 0, 0
    recalls, precisions, f1s = [], [], []
    # Load gold_answers and predctions
    gold_answers = load_gt(ok_vqa_gt_file)

    if use_descriptions:
        llava_answers = load_llaava_output(llava_output_file)
    else:
        llava_answers = load_llaava_output(llava_output_single_word_file)
    
    for line in llava_answers:
        question_id = line['question_id']
        question = line['prompt'].split("\n")[0]
        llava_answer = line['text']
        gold_answer = gold_answers[question_id]
        # print(f'Question: {question} \nGold: {gold_answer} \nLLAVA: {llava_answer}')
        
        if exact_matching:
            matches, match_count, num_gold, match_details = do_exact_matching(gold_answer, llava_answer)
        else:
            # matches, match_count, num_gold = get_num_matches(gold_answer, llava_answer)
            matches, match_count, num_gold, match_details = get_num_matches_spacy(gold_answer, llava_answer)
        
        if debug_wrong_predictions: 
            if not num_gold:
                print(f'\nQuestion: {question} \nGold: {gold_answer} \nLLAVA: {llava_answer}')
        
        # print(f'Matches: {matches} Match Details: {match_details} Match count: {match_count} Gold count: {num_gold}/{len(gold_answer)}')
        # print("\n")
        llava_answer_words = llava_answer.split()
        if len(matches):
            f1_curr, curr_precision, curr_recall = f1(matches[0], gold_answer, tokenizer)
        else:
            f1_curr, curr_precision, curr_recall = f1(llava_answer_words[0], gold_answer, tokenizer)
        # print(f'F1: {f1_curr:.2f}, Precision: {curr_precision:.2f}, Recall: {curr_recall:.2f}')
        total_match_count += num_gold
        total_gold_count += len(gold_answer)
        if num_gold > 0:
            atleast_one_match += 1
        f1s.append(f1_curr)
        recalls.append(curr_recall)
        precisions.append(curr_precision)
        
        # cnt += 1
        # if cnt > 10:
        #     break
    
    F1_score = sum(f1s) / len(f1s)
    recall = sum(recalls) / len(recalls)
    precision = sum(precisions) / len(precisions)
    print('\n\n')
    print(f'Total match count: {total_match_count}, Total gold count: {total_gold_count}, Atleast one match: {atleast_one_match}')
    print(f'Accuracy: {total_match_count/total_gold_count*100:.2f}%')
    print(f'Atleast one match accuracy: {atleast_one_match*100/len(llava_answers):.2f}%')
    print(f'F1: {F1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')



