# CC3 - Deep Learning Model Comparison

## Project
Streamlit app comparing fine-tuned YOLO and ViT models on Cats vs Dogs dataset.

## Installation
pip install -r requirements.txt

## Run Fine-tuning
python src/train_vit.py
python src/train_yolo.py

## Launch App
streamlit run app.py

## Models
- YOLOv8: Object detection
- ViT: Image classification

## Dataset
150 images per class (cats/dogs), split 60/20/20
