# Fine-tuning ViT (Vision Transformer) for cats vs dogs classification
import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from torch.optim import Adam
import json, time, os
import matplotlib.pyplot as plt

print("Loading ViT model...")

# Settings
EPOCHS = 5
BATCH  = 16
LR     = 2e-5
DEVICE = "cpu"  # no GPU, we use CPU

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Load images from folders
train_data = datasets.ImageFolder("data/train", transform=transform)
val_data   = datasets.ImageFolder("data/val",   transform=transform)
test_data  = datasets.ImageFolder("data/test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH)
test_loader  = DataLoader(test_data,  batch_size=BATCH)

print(f"Classes: {train_data.classes}")

# Load pretrained ViT and adapt to 2 classes
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model = model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.CrossEntropyLoss()

train_losses, val_losses, val_accs = [], [], []

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images).logits
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

training_time = time.time() - start_time
print(f"\nTraining done in {training_time:.1f} seconds")

# Save model
os.makedirs("models/vit_finetuned", exist_ok=True)
model.save_pretrained("models/vit_finetuned")
print("Model saved!")

# Evaluate on test set
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(pixel_values=images).logits
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(all_labels, all_preds, average="weighted")
recall    = recall_score(all_labels, all_preds, average="weighted")
f1        = f1_score(all_labels, all_preds, average="weighted")
accuracy  = correct / total

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

# Save metrics
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "training_time_seconds": round(training_time, 1),
    "epochs": EPOCHS
}
with open("models/vit_finetuned/metrics_vit.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved!")

# Save learning curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.title("ViT - Loss Curves")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label="Val Accuracy", color="green")
plt.title("ViT - Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.savefig("models/vit_finetuned/training_curves.png")
print("Curves saved!")