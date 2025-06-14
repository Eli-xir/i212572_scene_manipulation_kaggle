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

def get_yolo_detections(image: Image.Image, target_classes=None, conf_threshold=0.5,yolo = None):
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

def process_images(image_sources, target_classes=None, conf_threshold=0.5,predictor = None,yolo = None):
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
def visualize_relighting(original_image, relit_image, segmentation_results):
    """
    Visualize before and after relighting
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Relit
    axes[1].imshow(relit_image)
    axes[1].set_title("Relit")
    axes[1].axis('off')
    
    # Difference (enhanced)
    diff = np.abs(np.array(relit_image).astype(np.int16) - np.array(original_image).astype(np.int16))
    diff = (diff / np.max(diff) * 255).astype(np.uint8) if np.max(diff) > 0 else diff.astype(np.uint8)
    axes[2].imshow(diff)
    axes[2].set_title("Difference (Enhanced)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_lighting_variations(original_image, variations):
    """
    Show multiple lighting variations in a grid
    """
    num_vars = len(variations)
    cols = min(4, num_vars + 1)
    rows = (num_vars + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Original
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    # Variations
    for i, var in enumerate(variations):
        row = (i + 1) // cols
        col = (i + 1) % cols
        axes[row, col].imshow(var['image'])
        axes[row, col].set_title(var['name'])
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_vars + 1, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

    # ==================== RELIGHTING PIPELINE ====================
def estimate_lighting_direction(image_np, mask):
    """
    Estimate lighting direction from gradients in the masked region.
    Simple approach: use image gradients to find dominant light direction.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply mask
    masked_gray = gray * mask.astype(np.uint8)
    
    # Compute gradients
    grad_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Find dominant gradient direction (simplified)
    valid_pixels = mask > 0
    if np.sum(valid_pixels) > 0:
        mean_grad_x = np.mean(grad_x[valid_pixels])
        mean_grad_y = np.mean(grad_y[valid_pixels])
        
        # Convert to lighting direction (opposite of gradient)
        light_direction = np.array([-mean_grad_x, -mean_grad_y, 50])  # Assume some Z component
        light_direction = light_direction / (np.linalg.norm(light_direction) + 1e-8)
    else:
        light_direction = np.array([0.5, -0.5, 0.7])  # Default top-left lighting
    
    return light_direction

def generate_normal_map(mask, blur_kernel=5):
    """
    Generate a simple normal map from the mask.
    This is a simplified approach - in practice you'd want more sophisticated methods.
    """
    # Blur the mask to create smooth transitions
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0)
    
    # Compute gradients
    grad_x = cv2.Sobel(mask_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Create normal map (simplified)
    normal_map = np.zeros((*mask.shape, 3))
    normal_map[:, :, 0] = grad_x / 255.0  # X component
    normal_map[:, :, 1] = grad_y / 255.0  # Y component
    normal_map[:, :, 2] = 1.0             # Z component (pointing up)
    
    # Normalize
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = normal_map / (norm + 1e-8)
    
    return normal_map

def apply_relighting(image_np, mask, light_direction, intensity=1.0, ambient=0.3):
    """
    Apply relighting to the masked object.
    Args:
        image_np: Original image as numpy array
        mask: Binary mask of the object
        light_direction: 3D vector for light direction
        intensity: Light intensity multiplier
        ambient: Ambient light level
    Returns:
        Relit image
    """
    # Generate normal map from mask
    normal_map = generate_normal_map(mask)
    
    # Compute lighting
    # Dot product of normal and light direction
    lighting = np.sum(normal_map * light_direction, axis=2)
    lighting = np.maximum(lighting, 0)  # Only positive lighting
    
    # Add ambient lighting
    lighting = ambient + (1 - ambient) * lighting * intensity
    
    # Apply to image
    relit_image = image_np.copy().astype(np.float32)
    
    # Apply lighting only to masked region
    for c in range(3):  # RGB channels
        channel = relit_image[:, :, c]
        channel[mask > 0] = channel[mask > 0] * lighting[mask > 0]
        relit_image[:, :, c] = np.clip(channel, 0, 255)
    
    return relit_image.astype(np.uint8)

def relight_objects(image, segmentation_results, target_light_direction=None, intensity=1.2, ambient=0.2):
    """
    Relight all segmented objects in an image.
    Args:
        image: PIL Image
        segmentation_results: Results from segment_objects
        target_light_direction: Target lighting direction [x, y, z] or None for auto
        intensity: Lighting intensity
        ambient: Ambient light level
    Returns:
        Relit PIL Image
    """
    image_np = np.array(image)
    relit_image = image_np.copy()
    
    for result in segmentation_results:
        mask = result['mask']
        
        # Use provided direction or estimate from image
        if target_light_direction is not None:
            light_dir = np.array(target_light_direction)
            light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
        else:
            light_dir = estimate_lighting_direction(image_np, mask)
        
        # Apply relighting to this object
        object_relit = apply_relighting(relit_image, mask, light_dir, intensity, ambient)
        
        # Blend back into image (only the masked region)
        relit_image[mask > 0] = object_relit[mask > 0]
    
    return Image.fromarray(relit_image)

# ==================== ADVANCED RELIGHTING ====================
def create_lighting_variations(image, segmentation_results, num_variations=4):
    """
    Create multiple lighting variations of the same image.
    """
    variations = []
    
    # Different lighting directions
    light_directions = [
        [1, 0, 1],      # Right
        [-1, 0, 1],     # Left  
        [0, -1, 1],     # Top
        [0.7, -0.7, 1], # Top-right
    ]
    
    for i, light_dir in enumerate(light_directions[:num_variations]):
        relit = relight_objects(image, segmentation_results, light_dir, intensity=1.3, ambient=0.15)
        variations.append({
            'image': relit,
            'direction': light_dir,
            'name': f'Variation_{i+1}'
        })
    
    return variations

def enhance_realism(image_np, mask, shadow_strength=0.3, highlight_strength=0.2):
    """
    Add realistic shadows and highlights to improve realism.
    """
    # Create shadow map (simple approach)
    shadow_map = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    shadow_map = shadow_map / np.max(shadow_map) if np.max(shadow_map) > 0 else shadow_map
    
    # Apply shadows around the object
    shadow_region = (shadow_map > 0.1) & (mask == 0)
    
    enhanced_image = image_np.copy().astype(np.float32)
    
    # Add shadows
    enhanced_image[shadow_region] *= (1 - shadow_strength * shadow_map[shadow_region, np.newaxis])
    
    # Add highlights on the object
    highlight_region = mask > 0
    enhanced_image[highlight_region] *= (1 + highlight_strength * 0.5)
    
    return np.clip(enhanced_image, 0, 255).astype(np.uint8)
