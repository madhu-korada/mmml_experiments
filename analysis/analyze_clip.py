from PIL import Image
import requests
import torch
import json
from pycocotools.coco import COCO
from transformers import CLIPProcessor, CLIPModel

from llava_evaluate import load_gt, load_llaava_output
from analysis.count_no_objs import load_ann_file, get_objects_bboxes, get_objects_bboxes_all_images, get_image_info_from_index

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_category_names(coco_api, ann_file):
    categories = coco_api.loadCats(coco_api.getCatIds())
    category_names = [cat['name'] for cat in categories]
    return category_names

def get_image_info(line, coco_api, image_ids):
    question_id = line['question_id']
    image_id = image_ids[question_id]
    image_file = f'data/coco2014/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg'
    bboxes, object_types = get_objects_bboxes(coco_api, image_id)
    return image_file, image_id, bboxes, object_types

def clip_multiclass_classification(image, category_names, topk=5):
    category_names_clip = ["a photo of a " + obj for obj in category_names]
    inputs = clip_processor(text=category_names_clip, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_probs, max_indices = torch.topk(probs, topk)
    
    labels, scores = [], []
    for i in range(max_indices.size(1)):
        label, score = category_names[max_indices[0, i]], max_probs[0, i].item()
        # print(f"Label: {label}, score: {score}")
        labels.append(label)
        scores.append(score)
        
    return labels, scores

def calculate_accuracy(gold_labels, clip_labels):
    matched_labels = [label for label in clip_labels if label in gold_labels]
    accuracy = len(matched_labels) / len(gold_labels)
    return accuracy


if __name__ == "__main__":
    EVAL_MODE = "CLIP"
    DO_MODEL_INFERENCE = False
    
    ok_vqa_gt_file = 'data/gt_ok_vqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    ann_file = 'data/gt_ok_vqa/instances_val2014.json'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'
    clip_predictions_file = 'data/clip_predictions.jsonl'

    llava_answers = load_llaava_output(llava_output_single_word_file)
    gold_answers, image_ids = load_gt(ok_vqa_gt_file)
    coco_api = load_ann_file(ann_file)
    category_names = get_category_names(coco_api, ann_file)
    bboxes_list, object_types_list = get_objects_bboxes_all_images(coco_api, image_ids)
    if not DO_MODEL_INFERENCE:
        clip_predictions = load_llaava_output(clip_predictions_file)
        
    ctr = 0
    accuracy = 0
    
    for line in llava_answers:
        ctr += 1
        image_file, image_id, bboxes, object_types = get_image_info(line, coco_api, image_ids)
        image = Image.open(image_file)
        gt_unique_classes = list(set(object_types))
        
        # print("Image file: ", image_file)
        # print("GT Labels: ", gt_unique_classes)
        
        if EVAL_MODE == "CLIP":
            if DO_MODEL_INFERENCE:
                labels, scores = clip_multiclass_classification(image, category_names, topk=len(gt_unique_classes))
                print("CLIP Labels: ", labels)
                # Matched labels with GT labels
                matched_labels = [label for label in labels if label in gt_unique_classes]
                
                json_dict = {
                    "no": ctr,
                    "image_id": image_id,
                    "gt_labels": gt_unique_classes,
                    "clip_labels": labels,
                    "matched_labels": matched_labels,
                    "gt_unfiltered_labels": object_types,
                    "gt_bboxes": bboxes,
                }
                # write to file
                with open(clip_predictions_file, "a") as f:
                    f.write(json.dumps(json_dict))
                    f.write("\n")
                    
                # print("Matched Labels: ", matched_labels)
                print("No. of matched labels: ", len(matched_labels), "/", len(gt_unique_classes))
                print("--------------------------------------------------")
            else:
                matched_labels = clip_predictions[ctr-1]["matched_labels"]
                clip_labels = clip_predictions[ctr-1]["clip_labels"]
                # print("CLIP Labels: ", clip_labels)
                # print("No. of matched labels: ", len(matched_labels), "/", len(gt_unique_classes))
                # print("--------------------------------------------------")
                
                # Calculate accuracy
                accuracy += calculate_accuracy(gt_unique_classes, matched_labels) if len(gt_unique_classes) > 0 else 0
    
    print("--------------------------------------------------")
    print("Accuracy: ", accuracy/ctr)
    print("Total images processed: ", ctr)
    print("--------------------------------------------------")
        
        