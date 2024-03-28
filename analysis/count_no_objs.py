import os
import json
import numpy as np
import pycocotools.coco as coco
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

def annotate_image_with_bboxes(image_path, image_id, bboxes):
    """
    Annotates an image with bounding boxes of objects based on COCO dataset annotations.

    Parameters:
    - image_path: str, the path to the image file
    - image_id: int, the COCO image ID to annotate
    - annotation_file: str, path to the COCO annotation file (instances file)
    """
    
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        # The COCO bbox format is [x, y, width, height]
        # PIL expects the bounding box in the format [x0, y0, x1, y1]
        x0, y0, width, height = bbox
        x1 = x0 + width
        y1 = y0 + height
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    
    # Display the annotated image
    image.show()
    
    # save the image
    image.save("annotated_image.jpg")
    return

def load_ann_file(annotation_file):
    """
    Returns a COCO api object for the given annotation file.
    
    Parameters:
    - annotation_file: str, path to the COCO annotation file (instances file)
    
    Returns:
    - coco: COCO api object
    """
    # Initialize COCO api for instance annotations
    coco = COCO(annotation_file)
    return coco

def get_objects_bboxes(coco, image_id):
    """
    Returns the number of objects and a list of object types for a given image ID.

    Parameters:
    - coco: COCO api object
    - image_id: int, the image ID to query
    
    Returns:
    - num_objects: int, the number of objects in the given image
    - object_types: list, a list of the types of objects in the image
    """
    # Get all annotations for the given image ID
    annIds = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(annIds)
    bboxes = [ann['bbox'] for ann in anns]
    
    # Count the number of objects and get their category names
    num_objects = len(bboxes)
    object_types = [coco.loadCats(ann['category_id'])[0]['name'] for ann in anns]
    
    return bboxes, object_types

def get_objects_bboxes_all_images(coco, image_ids):
    """
    Returns the number of objects and a list of object types for a given image ID.

    Parameters:
    - coco: COCO api object
    - image_ids: list, a list of image IDs to query

    Returns:
    - num_objects: int, the number of objects in the given image
    - object_types: list, a list of the types of objects in the image
    """
    
    bboxes_list = []
    object_types_list = []
    for image_id in image_ids:
        # Get all annotations for the given image ID
        annIds = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(annIds)
        bboxes = [ann['bbox'] for ann in anns]
        
        # Count the number of objects and get their category names
        num_objects = len(bboxes)
        object_types = [coco.loadCats(ann['category_id'])[0]['name'] for ann in anns]
        bboxes_list.append(bboxes)
        object_types_list.append(object_types)
    
    return bboxes_list, object_types_list

def get_image_info_from_index(i, image_ids, bboxes_list, object_types_list):
    image_id = image_ids[i]
    image_file = f'data/coco2014/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg'
    return image_file, image_id, bboxes_list[i], object_types_list[i]


if __name__ == "__main__":
    
    # 184613
    annotation_file = "data/gt_ok_vqa/instances_val2014.json" # Update this path
    image_id = 94922 # Replace with a valid image ID from your dataset

    coco = load_ann_file(annotation_file)
    
    bboxes, object_types = get_objects_bboxes(coco, image_id)
    
    image_path = "data/coco2014/val2014/COCO_val2014_000000094922.jpg"
    annotate_image_with_bboxes(image_path, image_id, bboxes)
    
    print(f"Number of objects: {len(bboxes)}")
    print(f"Object types: {object_types}")

