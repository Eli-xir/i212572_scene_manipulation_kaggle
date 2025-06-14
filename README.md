# Scene Manipulation with AI

An AI system that edits images using plain English instructions — move objects, change lighting, and apply visual effects automatically.

---

## What This Does

Use natural language instructions like:

* Move the car to the left
* Add sunset lighting
* Place the bird in the center

The system combines YOLOv8 (object detection), SAM (segmentation), Stable Diffusion (inpainting), and CLIP (quality scoring) to perform complex image manipulations.

---

## Quick Start

### Option 1: Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new Notebook
2. In **Notebook Settings**, enable **GPU (T4 x2)**
3. Install dependencies inside your notebook
4. Download dataset with:

```python
import kagglehub
path = kagglehub.dataset_download("elatedspider/scene-manipulation-dataset")
print("Dataset path:", path)

```

5. Import the given kaggle notebook to your current session state.
6. Click **Run All** and wait for model downloads (5–10 minutes on first run)

---

### Option 2: Local Setup

**Requirements**:

* Python 3.8+
* CUDA-capable GPU (8GB+ recommended)
* \~15GB free disk space

**Installation**:

```bash
git clone <your-repo-url>
cd scene-manipulation

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers ultralytics segment-anything-py
pip install opencv-python pillow matplotlib numpy spacy accelerate xformers

python -m spacy download en_core_web_sm
```

**Important**:

* Update image paths in the code to match your local setup
* SAM checkpoints must be downloaded manually (from Meta's SAM GitHub repo)
* Models will auto-download on first run

---

## Usage Example

```python
instruction = "Move the car to the left and add sunset lighting"
image_sources = [('url', 'your_image_url_here')]

results = complete_scene_manipulation_pipeline(
    image_sources=image_sources,
    instruction=instruction,
    nlp_model=nlp,
    predictor=predictor,
    yolo=yolo,
    target_classes=['car', 'truck', 'bus'],
    conf_threshold=0.7
)
```

---

## Supported Instructions

### Object Movement

* Move the \[object] to the \[left/right/center/top/bottom]
* Place the \[object] in the \[position]
* Relocate the \[object] \[direction]

### Lighting Effects

* Add \[sunset/sunrise/dramatic] lighting
* Make it \[brighter/darker]
* Add \[warm/cool] lighting

### Style Modifiers

* slightly, very, much (intensity)
* natural, dramatic, subtle (style)

---

## What You Get

* Before/After comparisons
* Segmentation masks
* CLIP similarity score
* Processing stats and timings
* Difference maps

---

## Common Issues

**Out of Memory**:

* Reduce image size or switch to CPU
* Close other GPU processes
* On Kaggle: make sure GPU is enabled

**Model Download Failed**:

* Check your internet
* Restart kernel (Kaggle)
* Clear HuggingFace cache (local)

**No Objects Detected**:

* Lower `conf_threshold` (try 0.3)
* Make sure the object class is included
* Use clear, visible images

**Path Issues (Local)**:

* Use absolute paths
* Ensure files exist and permissions are correct

---

## Tips for Best Results

* Use clear, well-lit images
* Objects should be visible, not occluded
* Higher resolution images work better (but slower)
* Keep instructions simple and direct
* Stick to known classes: car, truck, bus, motorcycle, bird, dog, bear

---

## Architecture

Natural Language → spaCy → YOLOv8 → SAM
→ Stable Diffusion (inpainting) → Object Relocation
→ CLIP scoring → Final output

---

## Performance

* Object detection: \~95% accuracy for common classes
* Processing time: 15–30 seconds/image (with GPU)
* Memory: 8–12GB VRAM
* CLIP similarity score: \~0.7–0.9 for good edits
