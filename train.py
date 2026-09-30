from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import DefectDetectorCNN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "dataset"
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "defect_detector.pt"
META_PATH = ARTIFACTS / "model_meta.json"
IMAGE_SIZE = 128
CLASSES = ["normal", "defective"]


class FolderDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, name in enumerate(CLASSES):
            for path in sorted((Path(root) / name).glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor, label


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, labels)) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return {"loss": loss_total / max(total, 1), "accuracy": correct / max(total, 1)}


def train(epochs=8, batch_size=16, learning_rate=1e-3):
    torch.manual_seed(42)
    dataset = FolderDataset(DATA_DIR)
    validation_size = max(2, int(round(len(dataset) * 0.2)))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectDetectorCNN(len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
        train_metrics = {"loss": loss_total / max(total, 1), "accuracy": correct / max(total, 1)}
        validation_metrics = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["accuracy"] >= best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "image_size": IMAGE_SIZE}, MODEL_PATH)
    META_PATH.write_text(json.dumps({"device": str(device), "classes": CLASSES, "image_size": IMAGE_SIZE, "history": history}, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
