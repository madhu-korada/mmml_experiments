import os
import re
import json
import nltk
import torch
import spacy
from fuzzywuzzy import fuzz
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import transformers

from f1_score import f1

nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def load_prophet_output(file_path):
    with open(file_path, 'r') as input_file:
        prophet_output = json.load(input_file)
        prophet_output = [x for x in prophet_output]
    return prophet_output

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

if __name__ == "__main__":
    use_descriptions = True
    prophet_json_file = 'data/eval_prophet/result_20240228154916.json'
    ok_vqa_gt_file = 'prophet/datasets/okvqa/mscoco_val2014_annotations.json'
    model_id = "gg-hf/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    cnt = 1
    # Load gold_answers and predctions
    gold_answers = load_gt(ok_vqa_gt_file)
    prophet_answers = load_prophet_output(prophet_json_file)

    total_match_count, total_gold_count, atleast_one_match = 0, 0, 0
    recalls, precisions, f1s = [], [], []

    for line in prophet_answers:
        question_id = line['question_id']
        # question = line['prompt'].split("\n")[0]
        prophet_answer = line['answer']
        gold_answer = gold_answers[question_id]
        
        # matches, match_count, num_gold = get_num_matches(gold_answer, prophet_answer)
        matches, match_count, num_gold, match_details = get_num_matches_spacy(gold_answer, prophet_answer)
        # if num_gold < 5:
        print(f'Gold: {gold_answer} \nProphet: {prophet_answer}')
        
        print(f'Matches: {matches} Match Details: {match_details} Match count: {match_count} Gold count: {num_gold}/{len(gold_answer)}')
        
        if len(matches) == 1:
            prophet_answer_words = matches[0]    
        elif len(matches) > 1:
            prophet_answer_words = matches[0]
        else:
            prophet_answer_words = prophet_answer.split()
            prophet_answer_words = prophet_answer_words[0]
        print(f'Prophet Answer Words: {prophet_answer_words}')            
        f1_curr, curr_precision, curr_recall = f1(prophet_answer_words, gold_answer, tokenizer)
        print(f'F1: {f1_curr:.2f}, Precision: {curr_precision:.2f}, Recall: {curr_recall:.2f}')
        total_match_count += num_gold
        total_gold_count += len(gold_answer)
        if num_gold > 0:
            atleast_one_match += 1
        f1s.append(f1_curr)
        recalls.append(curr_recall)
        precisions.append(curr_precision)
        print("\n")
        
        cnt += 1
        if cnt > 1000:
            break

    F1_score = sum(f1s) / len(f1s)
    recall = sum(recalls) / len(recalls)
    precision = sum(precisions) / len(precisions)

    print(f'Total match count: {total_match_count}, Total gold count: {total_gold_count}, Atleast one match: {atleast_one_match}')
    print(f'Accuracy: {total_match_count/total_gold_count*100:.2f}%')
    print(f'Atleast one match accuracy: {atleast_one_match*100/cnt:.2f}%')
    print(f'F1: {F1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
