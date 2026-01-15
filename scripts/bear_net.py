from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from PIL import Image
import os
import yaml
import json

with open('dvc_files/params.yaml', 'r') as file:
    params=yaml.safe_load(file)

BATCH_SIZE = params['train']['batch_size']
EPOCHS = params['train']['epochs']
LR=params['train']['lr']
WD=params['train']['weight_decay']
TEST_SIZE=params['data']['subset']
random_state=params['data']['random_state']

IMAGE_SIZE = 224
DEVICE = "cpu"
path="bear_dataset"
print(path)

full_dataset = datasets.ImageFolder(root=path, transform=None)
paths = [sample[0] for sample in full_dataset.samples]   # пути к изображениям
labels = [sample[1] for sample in full_dataset.samples]  # метки класса

print("Классы:", full_dataset.classes)
print("Распределение классов:", torch.bincount(torch.tensor(labels)).tolist())

train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=TEST_SIZE,
                                                                      stratify=labels, random_state=random_state)

print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"Ошибка загрузки {self.image_paths[idx]}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

train_dataset = ImagePathDataset(train_paths, train_labels, transform=transforms_train)
test_dataset = ImagePathDataset(test_paths, test_labels, transform=transforms_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b0(weights=None)
state_dict = torch.load("efficientnet_b0_rwightman-3dd342df.pth", map_location="cpu")
model.load_state_dict(state_dict)

for p in model.features.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

if EPOCHS!=0:
    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        for x, y in tqdm(train_loader):
          x, y = x.to(DEVICE), y.to(DEVICE)

          optimizer.zero_grad()
          out = model(x)
          loss = criterion(out, y)
          loss.backward()
          optimizer.step()

          pred = out.argmax(1)
          correct += (pred == y).sum().item()
          total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: train accuracy = {acc:.4f}")

    st_after_training=model.state_dict()
    torch.save(st_after_training,'model_bear.tar')
    print("Модель сохранена как 'model_bear.tar'")

if EPOCHS==0:
    st_loaded=torch.load('model_bear.tar', weights_only=True)
    model.load_state_dict(st_loaded)

model.eval()
if EPOCHS!=0:
    total, correct = 0, 0
    with torch.no_grad():
      for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    val_acc = correct / total
    print(f"\nИтоговая точность на тесте: {val_acc:.4f}")

metrics = {
    "test_accuracy": float(val_acc),
    "train_accuracy_last_epoch": float(acc) if EPOCHS > 0 else None,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}

with open("dvc_files/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

#in my prev test i have train accuracy 0.8141,  test accuracy = 0.8316

all_paths = [sample[0] for sample in full_dataset.samples]
all_labels = [sample[1] for sample in full_dataset.samples]

all_dataset = ImagePathDataset(all_paths, all_labels, transform=transforms_test)
all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False)

embedding_model = nn.Sequential( model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(DEVICE)
embeddings = []
labels = []
urls = []
embedding_model.eval()
with torch.no_grad():
  for x, y in tqdm(all_loader):
    emb = embedding_model(x)
    embeddings.append(emb.numpy().astype("float16"))
    labels.extend(y.numpy())
embeddings = np.vstack(embeddings)
labels = np.array(labels)

all_paths_abs = np.array([os.path.abspath(p) for p in all_paths])

np.savez_compressed("bear_embeddings_all.npz", embeddings=embeddings,
                    labels=labels, image_paths=all_paths_abs, class_names=np.array(full_dataset.classes))

st_dict_emb=model.state_dict()
torch.save(st_dict_emb,'embedding_model_bear_1.tar')
print("Модель сохранена как 'embedding_model_bear_1.tar'")

data = np.load("bear_embeddings_all.npz", allow_pickle=True)
loaded_embeddings = data["embeddings"]
loaded_labels = data["labels"]
loaded_paths = data["image_paths"]
loaded_classes = data["class_names"]
print(f"Загружено: {loaded_embeddings.shape}, метки: {loaded_labels.shape}")

knn_loaded = NearestNeighbors(n_neighbors=4, metric="cosine")
knn_loaded.fit(loaded_embeddings)

def predict_and_show(image_path, emb_file="loaded/bear_embeddings_all.npz"):
    data = np.load(emb_file, allow_pickle=True)
    knn = knn_loaded
    class_names = data["class_names"]

    image = Image.open(image_path).convert("RGB")
    x = transforms_test(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred_class = logits.argmax(1).item()
        emb = embedding_model(x).cpu().numpy()
    distances, indices = knn.kneighbors(emb)
    print(f"Предсказанный класс: {class_names[pred_class]}")
    print("3 похожих изображения из обучающей выборки:")
    for i, idx in enumerate(indices[0][1:4], 1):
        label = data["labels"][idx]
        path = data["image_paths"][idx]
        print(f"  {i}. {os.path.basename(path)} → {class_names[label]} (dist={distances[0][i]:.3f})")

predict_and_show('examples/white_bear_example.jpg')
