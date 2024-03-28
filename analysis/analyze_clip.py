from PIL import Image
import requests
import torch
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

if __name__ == "__main__":
    EVAL_MODE = "CLIP"
    
    ok_vqa_gt_file = 'data/gt_ok_vqa/mscoco_val2014_annotations.json'
    llava_output_file = 'data/eval_llava/answer-file-our.jsonl'
    ann_file = 'data/gt_ok_vqa/instances_val2014.json'
    llava_output_single_word_file = 'data/eval_llava/llava-1.6-answer-file-our-single-word-temp-1-beams-5.jsonl'

    llava_answers = load_llaava_output(llava_output_single_word_file)
    gold_answers, image_ids = load_gt(ok_vqa_gt_file)
    coco_api = load_ann_file(ann_file)
    category_names = get_category_names(coco_api, ann_file)
    bboxes_list, object_types_list = get_objects_bboxes_all_images(coco_api, image_ids)
    
    
    for line in llava_answers:
        image_file, image_id, bboxes, object_types = get_image_info(line, coco_api, image_ids)
        image = Image.open(image_file)
        gt_unique_classes = list(set(object_types))
        
        print("Image file: ", image_file)
        print("Image ID: ", image_id)
        print("Bounding boxes: ", bboxes)
        print("GT Labels: ", gt_unique_classes)
        
        if EVAL_MODE == "CLIP":
            labels, scores = clip_multiclass_classification(image, category_names, topk=len(unique_classes))
            print("CLIP Labels: ", labels)
            # print("CLIP Scores: ", scores)
        
        exit()
        
        