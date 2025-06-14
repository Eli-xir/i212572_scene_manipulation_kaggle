import spacy

nlp = spacy.load("en_core_web_sm")

def parse_instruction(instruction: str) -> dict:
    doc = nlp(instruction.lower())

    action_details = {
        "object": None,
        "relocation": None,
        "relighting": None
    }

    # Step 1: Find object of motion verb
    for token in doc:
        if token.dep_ in ("dobj", "pobj") and token.head.lemma_ in ["move", "place", "put", "relocate"]:
            action_details["object"] = token.text
            break  # Stop after finding the first good object

    # Step 2: Fallback to first noun chunk
    if not action_details["object"]:
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ != "nsubj":
                action_details["object"] = chunk.root.text
                break

    # Step 3: Check for relocation cues
    for token in doc:
        if token.text in ["left", "right", "top", "bottom", "center"]:
            action_details["relocation"] = token.text
            break

    # Step 4: Check for relighting phrases
    for token in doc:
        if "light" in token.text or "shadow" in token.text or "sun" in token.text:
            subtree = list(token.head.subtree)
            lighting_phrase = [t.text for t in subtree if t.pos_ in ["ADJ", "NOUN"]]
            action_details["relighting"] = " ".join(lighting_phrase)
            break

    return action_details

#================================== Yolo and SAM model helper functions ====================================
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import requests
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def save_masks(results, output_dir='masks'):
    Path(output_dir).mkdir(exist_ok=True)
    
    for img_idx, result in enumerate(results):
        for obj_idx, obj in enumerate(result['results']):
            mask_path = f"{output_dir}/img_{img_idx}_obj_{obj_idx}_{obj['class']}.png"
            mask_img = Image.fromarray((obj['mask'] * 255).astype(np.uint8))
            mask_img.save(mask_path)
            print(f"Saved mask: {mask_path}")

def get_object_crops(image, results):
    crops = []
    image_np = np.array(image)
    
    for result in results:
        mask = result['mask']
        bbox = result['bbox']
        
        # Apply mask to image
        masked_img = image_np.copy()
        masked_img[~mask] = 0  # Zero out non-object pixels
        
        # Crop to bounding box
        crop = masked_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crops.append({
            'crop': Image.fromarray(crop),
            'class': result['class'],
            'confidence': result['confidence']
        })
    
    return crops

def load_images(image_sources):
    """
    Load images from URLs or local paths.
    Args:
        image_sources: List of tuples [(source_type, path), ...]
                      source_type: 'url' or 'local'
    Returns:
        List of PIL Images
    """
    images = []
    for source_type, path in image_sources:
        if source_type == 'url':
            response = requests.get(path, stream=True)
            img = Image.open(response.raw).convert("RGB")
        else:  # local
            img = Image.open(path).convert("RGB")
        images.append(img)
    return images

def get_yolo_detections(image: Image.Image, target_classes=None, conf_threshold=0.5,yolo = YOLO('yolo8n.pt')):
    """
    Get YOLO detections for an image.
    Args:
        image: PIL Image
        target_classes: List of class names to filter (None for all)
        conf_threshold: Confidence threshold
    Returns:
        List of bounding boxes and class info
    """
    results = yolo(image, conf=conf_threshold)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                if target_classes is None or class_name in target_classes:
                    detections.append({
                        'bbox': bbox.astype(int),
                        'class': class_name,
                        'confidence': confidence
                    })
    
    return detections

def segment_objects(image: Image.Image, detections,predictor = None):
    """
    Generate SAM masks for detected objects.
    Args:
        image: PIL Image
        detections: List of detection dictionaries from YOLO
    Returns:
        List of masks and detection info
    """
    image_np = np.array(image)
    predictor.set_image(image_np)
    
    results = []
    for detection in detections:
        bbox = detection['bbox']
        
        masks, scores, logits = predictor.predict(
            box=bbox[None, :],
            multimask_output=True,
        )
        
        # Take the mask with highest score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        results.append({
            'mask': best_mask,
            'bbox': bbox,
            'class': detection['class'],
            'confidence': detection['confidence'],
            'mask_score': scores[best_mask_idx]
        })
    
    return results

def process_images(image_sources, target_classes=None, conf_threshold=0.5,predictor = None,yolo = YOLO('yolo8n.pt')):
    """
    Full pipeline: load images -> detect objects -> segment objects
    """
    images = load_images(image_sources)
    all_results = []
    
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        
        # Get YOLO detections
        detections = get_yolo_detections(image, target_classes, conf_threshold,yolo)
        print(f"Found {len(detections)} objects")
        
        # Generate SAM masks
        segmentation_results = segment_objects(image, detections,predictor)
        
        all_results.append({
            'image': image,
            'results': segmentation_results
        })
    
    return all_results

def visualize_results(image, results, show_boxes=True, show_masks=True):
    """
    Visualize detection and segmentation results
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Detections with boxes
    img_with_boxes = np.array(image)
    for result in results:
        if show_boxes:
            bbox = result['bbox']
            cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{result['class']} {result['confidence']:.2f}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    axes[1].imshow(img_with_boxes)
    axes[1].set_title("YOLO Detections")
    axes[1].axis('off')
    
    # Combined masks
    if show_masks and results:
        combined_mask = np.zeros_like(results[0]['mask'], dtype=float)
        for i, result in enumerate(results):
            combined_mask += result['mask'] * (i + 1)
        
        axes[2].imshow(combined_mask, cmap='tab10')
        axes[2].set_title("SAM Segmentation Masks")
        axes[2].axis('off')
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()