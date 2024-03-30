from PIL import Image
import requests
import torch
import json
from pycocotools.coco import COCO
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llava_evaluate import load_gt, load_llaava_output, do_exact_matching, get_num_matches, get_num_matches_spacy
from analysis.count_no_objs import load_ann_file, get_objects_bboxes, get_objects_bboxes_all_images, get_image_info_from_index
from f1_score import calculate_f1_score, f1
from prophet_evaluate import load_prophet_output, load_questions

def read_question_types_file(question_types_file):
    with open(question_types_file, 'r') as f:
        question_types = f.readlines()
    return question_types

def get_question_types(question_types_file):
    lines = read_question_types_file(question_types_file)
    question_types = {}
    for line in lines:
        if line.endswith(":\n"):
            question_type = line.strip()[:-1]
            question_types[question_type] = []
        else:
            question_types[question_type].append(line.strip())
    return question_types
    
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
    EVAL_MODE = "CLIP"
    # EVAL_MODE = "PROPHET"
    DO_MODEL_INFERENCE = False
    exact_matching = False
    debug = False
    
    ok_vqa_gt_file = 'data/gt_ok_vqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    ann_file = 'data/gt_ok_vqa/instances_val2014.json'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'
    prophet_json_file = 'data/eval_prophet/result_20240228154916.json'
    question_types_file = 'data/questions_types.txt'

    model_id = "gg-hf/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    gold_answers, image_ids = load_gt(ok_vqa_gt_file)
    llava_answers = load_llaava_output(llava_output_single_word_file)
    prophet_answers = load_prophet_output(prophet_json_file)
    prophet_questions = load_questions()
    
    question_types = get_question_types(question_types_file)    
    print("Question Types: ", question_types.keys())
    
    acc_question_type = {}
    
    for question_type, questions in question_types.items():
        print(f'--------------------------------------------------')
        print(f"Question Type: {question_type}")
        cnt = 1
        total_match_count, total_gold_count, atleast_one_match = 0, 0, 0
        recalls, precisions, f1s = [], [], []
    
        for question in questions:
            question_id = get_qid_given_question(question, prophet_questions)
            question = prophet_questions[question_id]
            line = get_line_given_qid(question_id, prophet_answers)
            prophet_answer = line['answer']
            gold_answer = gold_answers[question_id]
            
            if exact_matching:
                matches, match_count, num_gold, match_details = do_exact_matching(gold_answer, prophet_answer)
            else:
                # matches, match_count, num_gold = get_num_matches(gold_answer, prophet_answer)
                matches, match_count, num_gold, match_details = get_num_matches_spacy(gold_answer, prophet_answer)
            
            prophet_answer_words = prophet_answer.split()
            if len(matches):
                f1_curr, curr_precision, curr_recall = f1(matches[0], gold_answer, tokenizer)
            else:
                f1_curr, curr_precision, curr_recall = f1(prophet_answer_words[0], gold_answer, tokenizer)
            # print(f'F1: {f1_curr:.2f}, Precision: {curr_precision:.2f}, Recall: {curr_recall:.2f}')
            total_match_count += num_gold
            total_gold_count += len(gold_answer)
            if num_gold > 0:
                atleast_one_match += 1
            f1s.append(f1_curr)
            recalls.append(curr_recall)
            precisions.append(curr_precision)
            
            cnt += 1
            if debug and cnt > 10:
                break
        
        F1_score = sum(f1s) / len(f1s)
        recall = sum(recalls) / len(recalls)
        precision = sum(precisions) / len(precisions)
        print(f'Total match count: {total_match_count}, Total gold count: {total_gold_count}, Atleast one match: {atleast_one_match}')
        print(f'Accuracy: {total_match_count/total_gold_count*100:.2f}%')
        print(f'Atleast one match accuracy: {atleast_one_match*100/len(questions):.2f}%')
        print(f'F1: {F1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
        print('\n\n')
        
        acc_question_type[question_type] = [len(questions), atleast_one_match*100/len(questions)]
    
    print(f'--------------------------------------------------')
    print(f'Accuracy for each question type:')
    for question_type, acc in acc_question_type.items():
        print(f'{question_type}: {acc[0]}, {acc[1]:.2f}%')
    print(f'--------------------------------------------------')
    
    # print in a file
    with open('data/question_types_accuracy_prophet.txt', 'w') as f:
        for question_type, acc in acc_question_type.items():
            f.write(f'{question_type}: {acc[0]}, {acc[1]:.2f}%\n')



