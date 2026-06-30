# Fine-tuning YOLOv8 for cats vs dogs detection
from ultralytics import YOLO
import json, time, os, shutil
import matplotlib.pyplot as plt

# First, let's create YOLO format dataset structure
os.makedirs("data/yolo/train/images", exist_ok=True)
os.makedirs("data/yolo/train/labels", exist_ok=True)
os.makedirs("data/yolo/val/images",   exist_ok=True)
os.makedirs("data/yolo/val/labels",   exist_ok=True)
os.makedirs("data/yolo/test/images",  exist_ok=True)
os.makedirs("data/yolo/test/labels",  exist_ok=True)

# Copy images and create label files
# cats = class 0, dogs = class 1
def prepare_yolo_split(split):
    for label, class_id in [("cats", 0), ("dogs", 1)]:
        src_folder = f"data/{split}/{label}"
        if not os.path.exists(src_folder):
            continue
        for img_file in os.listdir(src_folder):
            if not img_file.endswith(".jpg"):
                continue
            # Copy image
            shutil.copy(
                f"{src_folder}/{img_file}",
                f"data/yolo/{split}/images/{label}_{img_file}"
            )
            # Create label file (bbox covering whole image: x_center=0.5, y_center=0.5, w=1.0, h=1.0)
            label_file = f"data/yolo/{split}/labels/{label}_{img_file.replace('.jpg', '.txt')}"
            with open(label_file, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("Preparing YOLO dataset...")
prepare_yolo_split("train")
prepare_yolo_split("val")
prepare_yolo_split("test")
print("YOLO dataset ready!")

# Create dataset YAML config file
yaml_content = f"""
path: {os.path.abspath("data/yolo")}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['cats', 'dogs']
"""
with open("data/yolo/dataset.yaml", "w") as f:
    f.write(yaml_content)
print("YAML config saved!")

# Load pretrained YOLOv8 nano (smallest and fastest)
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Fine-tune
print("Starting YOLO training...")
start_time = time.time()

results = model.train(
    data="data/yolo/dataset.yaml",
    epochs=10,
    imgsz=224,
    batch=16,
    device="cpu",
    project="models/yolo_finetuned",
    name="train",
    exist_ok=True,
    verbose=True
)

training_time = time.time() - start_time
print(f"Training done in {training_time:.1f} seconds")

# Evaluate on test set
print("Evaluating on test set...")
metrics = model.val(data="data/yolo/dataset.yaml", split="test")

# Save metrics
m = {
    "precision": round(float(metrics.box.mp), 4),
    "recall": round(float(metrics.box.mr), 4),
    "f1": round(float(metrics.box.f1.mean()), 4),
    "mAP50": round(float(metrics.box.map50), 4),
    "training_time_seconds": round(training_time, 1),
    "epochs": 10
}
with open("models/yolo_finetuned/metrics_yolo.json", "w") as f:
    json.dump(m, f, indent=2)

print("\nYOLO Metrics:")
for k, v in m.items():
    print(f"  {k}: {v}")
print("Metrics saved!")