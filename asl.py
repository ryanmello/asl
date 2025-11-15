import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

DATA_DIR = "./asl-alphabet"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    print(f" Using device: {DEVICE}")

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "asl_alphabet_train/asl_alphabet_train"), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "asl_alphabet_test"), transform=val_transforms)

    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True  # Faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    print(f"Number of training images: {len(train_data)}")
    print(f"Number of validation images: {len(val_data)}")

    num_classes = len(train_data.classes)
    print(f"Number of classes detected: {num_classes}")
    print(f"Classes: {train_data.classes}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    from torch.amp import autocast, GradScaler

    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=use_cuda)

    train_acc_list, val_acc_list, train_loss_list = [], [], []

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        running_loss_sum = 0.0
        running_samples = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # weighted loss accumulation (per-sample)
            bs = labels.size(0)
            running_loss_sum += loss.item() * bs
            running_samples += bs

            # accuracy (every batch, cheap)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += bs

        epoch_loss = running_loss_sum / running_samples
        train_loss_list.append(epoch_loss)
        train_acc = 100.0 * correct / total
        train_acc_list.append(train_acc)

        # ---- VALIDATION ----
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                if use_cuda:
                    with autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs
                val_samples += bs

                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += bs

        val_loss = val_loss_sum / val_samples
        val_acc = 100.0 * val_correct / val_total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training completed!")

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_data.classes
    }, "sign_language_resnet18.pth")

    print("Model saved as sign_language_resnet18.pth")

    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()
