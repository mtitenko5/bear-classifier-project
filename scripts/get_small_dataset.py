import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

EMBEDDINGS_FILE = "bear_embeddings_all.npz"
OUTPUT_DIR = "bear_dataset_mini"
MAX_PER_CLASS = 100
TARGET_QUALITY = 75  # качество JPEG
MAX_TOTAL_IMAGES = 800

data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)  # [N, D]
labels = data["labels"]                             # [N]
image_paths = data["image_paths"]                   # [N]
class_names = data["class_names"]                   # [num_classes]

print(f"Всего изображений: {len(embeddings)}")
print(f"Классы: {class_names}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_copied = 0
for class_id, class_name in enumerate(class_names):
    print(f"\nОбрабатываем класс: {class_name} (ID={class_id})")
    
    idxs = np.where(labels == class_id)[0]
    n_select = min(MAX_PER_CLASS, len(idxs))
    print(f"  Всего изображений: {len(idxs)}, выбираем: {n_select}")

    if len(idxs) <= MAX_PER_CLASS:
        selected_idxs = idxs
    else:
        X = embeddings[idxs]
        print(f"  Кластеризация {len(X)} эмбеддингов...")
        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        selected_idxs = []
        for cluster_id in range(n_select):
            cluster_member_idxs = idxs[clusters == cluster_id]
            if len(cluster_member_idxs) > 0:
                chosen = np.random.choice(cluster_member_idxs)
                selected_idxs.append(chosen)
            else:
                selected_idxs.append(idxs[0])
        selected_idxs = np.array(selected_idxs)

    class_output_dir = os.path.join(OUTPUT_DIR, str(class_name))
    os.makedirs(class_output_dir, exist_ok=True)

    for i, idx in enumerate(tqdm(selected_idxs, desc=f"  Сохранение {class_name}")):
        src_path = image_paths[idx]
        if not os.path.exists(src_path):
            print(f"Файл не найден: {src_path}")
            continue
        try:
            img = Image.open(src_path).convert("RGB")
            dst_path = os.path.join(class_output_dir, f"{i:04d}.jpg")
            img.save(dst_path, "JPEG", quality=TARGET_QUALITY, optimize=True)
            total_copied += 1
        except Exception as e:
            print(f"Ошибка при обработке {src_path}: {e}")

    print(f" Сохранено {len(selected_idxs)} изображений")

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

DEVICE = "cpu"
BATCH_SIZE=32
path="data/bear_dataset_mini"
full_dataset = datasets.ImageFolder(root=path, transform=None)

paths = [sample[0] for sample in full_dataset.samples]
labels = [sample[1] for sample in full_dataset.samples]

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

all_dataset = ImagePathDataset(paths, labels, transform=transforms)
all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b0(weights=None)
st_loaded=torch.load("efficientnet_b0_rwightman-3dd342df.pth", map_location="cpu")
model.load_state_dict(st_loaded)

embedding_model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(DEVICE)
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

all_paths_abs = np.array([os.path.abspath(p) for p in paths])

np.savez_compressed("embeddings_mini.npz", embeddings=embeddings,
                    labels=labels, image_paths=all_paths_abs, class_names=np.array(full_dataset.classes))
