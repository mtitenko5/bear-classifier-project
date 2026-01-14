import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

DEVICE = "cpu"
MODEL_PATH = "loaded/model_bear.tar"
EMBEDDINGS_PATH = "loaded/embeddings_mini.npz"

bears_name_russian = {
    "Ursus americanus": "Американский чёрный медведь",
    "Ursus arctos": "Бурый медведь",
    "Ursus maritimus": "Белый медведь",
    "Ailuropoda melanoleuca": "Большая панда",
    "Ursus thibetanus": "Гималайский медведь",
    "Melursus ursinus": "Губач",
    "Helarctos malayanus": "Малайский медведь",
    "Tremarctos ornatus": "Очковый медведь"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model_and_embeddings():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"]      # [N, D]
    labels = data["labels"]              # [N]
    image_paths = data["image_paths"]    # [N] — строки
    class_names = data["class_names"].tolist()  # list of str

    knn = NearestNeighbors(n_neighbors=4, metric="cosine", n_jobs=-1)
    knn.fit(embeddings)
    return model, knn, embeddings, labels, image_paths, class_names

try:
    model, knn, embeddings, labels, image_paths, class_names = load_model_and_embeddings()
    st.sidebar.success(f" Загружено: {len(embeddings)} изображений")
except Exception as e:
    st.sidebar.error(f" Ошибка: {e}")
    st.stop()

st.set_page_config(page_title=" Bear Classifier", layout="centered")
st.title("Классификатор медведей")
st.markdown("Загрузите фото, модель определит вид медведя и покажет 3 похожих изображения из датасета.")

uploaded_file = st.file_uploader("Выберите изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ваше изображение")
    with st.spinner(" Анализируем..."):
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(1).item()
            conf = probs[0, pred_idx].item()
        pred_class = class_names[pred_idx]
        if pred_class in bears_name_russian:
            pred_class_ru = bears_name_russian[pred_class]

        embedding_model = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten() ).to(DEVICE)
        embedding_model.eval()
        with torch.no_grad():
            emb = embedding_model(x).cpu().numpy()

        distances, indices = knn.kneighbors(emb)
        top3_indices = indices[0][1:4]
        top3_distances = distances[0][1:4]

    st.subheader("Результат")
    st.markdown(f"**Вид медведя:** `{pred_class_ru}`")
    st.markdown(f"**Научное название:** `{pred_class}`")
    st.markdown(f"**Уверенность:** `{conf:.1%}`")

    st.subheader("Похожие изображения")
    cols = st.columns(3)
    for i, (col, idx) in enumerate(zip(cols, top3_indices)):
        path = image_paths[idx]
        true_label = labels[idx]
        true_class = class_names[true_label]
        dist = top3_distances[i]
        try:
            neighbor_img = Image.open(path).convert("RGB")
            neighbor_img.thumbnail((300, 300))
            with col:
                st.image(neighbor_img, caption=f"{true_class}\n(dist={dist:.3f})")
        except Exception as e:
            with col:
                st.error(f" {os.path.basename(path)}")

st.sidebar.markdown("---")
st.sidebar.title("Информация")
info_text = f"""\
- **Модель:** EfficientNet-B0  
- **Размер эмбеддингов:** {embeddings.shape[1]}  
- **Изображений:** {len(embeddings)}
- **Классов:** {len(class_names)}  
- **Классы:**  
"""
for cls in class_names:
    ru_name = bears_name_russian.get(cls, cls)
    info_text += f"  - {ru_name}\n"

st.sidebar.markdown(info_text)