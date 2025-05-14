import torch
from torch import nn
import os
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
PATCH_SIZE =16
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 12
DROPOUT = 0.1
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = 'gelu'
NUM_ENCODERS = 12
EMBED_DIM = 768
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
NUM_EPOCHS = 1000
GRADIENT_CLIP_NORM = 1.0

label_smoothing = 0.1
save_path = r'/kaggle/working/my_model.pth'
ckpt_path_temp = r"/kaggle/input/vit_plant_disease/transformers/default/1/my_model.pth"

test_data_path = r"/kaggle/input/plant-disease-recognition-dataset/Test/Test"
train_data_path = r"/kaggle/input/plant-disease-recognition-dataset/Train/Train"

load_checkpoint = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches,dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size),
            nn.Flatten(2)
            )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1,embed_dim)), requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0,2,1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_embeddings
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,dim_feedforward=4*embed_dim, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x

image_size = IMG_SIZE
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.Lambda(lambda img: img.convert('RGBA').convert('RGB') if img.mode == 'P' else img.convert('RGB')),

    transforms.ToTensor(),
    normalize,
])
test_transforms = transforms.Compose([
    transforms.Resize(int(image_size * 256 / 224)),
    transforms.CenterCrop(image_size),
    transforms.Lambda(lambda img: img.convert('RGBA').convert('RGB') if img.mode == 'P' else img.convert('RGB')),

    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_HEADS,DROPOUT, ACTIVATION, IN_CHANNELS).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            orig = model(x)
            flip = model(T.functional.hflip(x))
            scores = (orig + flip) / 2.0
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)
    model.train()
    return num_correct / num_samples

prev_test_acc = 0.0
if load_checkpoint and ckpt_path_temp.strip() and os.path.isfile(ckpt_path_temp.strip()):
    sd = torch.load(ckpt_path_temp.strip(), map_location=device)
    model.load_state_dict(sd.get('state_dict', sd), strict=False)
else:
    print("No checkpoint loaded; training from scratch")

print("Evaluating initial model accuracy...")
prev_test_acc = check_accuracy(test_loader, model)
print(f"Initial model accuracy (TTA): {prev_test_acc*100:.2f}%")
torch.save(model.state_dict(), save_path)
print(f"Saved initial model state to {save_path}")

print("Starting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    loop = tqdm(train_loader,
                desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training",
                leave=False)
    epoch_loss = 0.0
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        if GRADIENT_CLIP_NORM > 0:
             clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(
            loss=loss.item(),
            lr=optimizer.param_groups[0]['lr']
        )
    loop.close()
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"\n--- Evaluating Epoch {epoch+1} ---")
    print(f"Average Training Loss: {avg_epoch_loss:.4f}")
    test_acc = check_accuracy(test_loader, model)
    print(f"Accuracy on test set (TTA): {test_acc*100:.2f}%")
    scheduler.step()
    if test_acc > prev_test_acc:
        print(f"*** New best test accuracy: {test_acc*100:.2f}% ***")
        torch.save(model.state_dict(), save_path)
        prev_test_acc = test_acc
    print("-" * 50)

print("\n--- Training Finished ---")
print("Final performance:")
if os.path.isfile(save_path):
    print(f"Loading best model from {save_path}")
    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
    except:
        print("Failed to load best model; using last state")
else:
    print("Best model not found; using last state")

model.eval()
final_train_acc = check_accuracy(train_loader, model)
final_test_acc  = check_accuracy(test_loader, model)
print(f"Final training acc (TTA): {final_train_acc*100:.2f}%")
print(f"Final test acc (TTA):    {final_test_acc*100:.2f}%")
print(f"Best test acc (TTA):     {prev_test_acc*100:.2f}%")
if prev_test_acc < 0.96:
    print("\nNote: Best test acc below target of 96%. Consider revisiting hyperparams or checkpoint.")