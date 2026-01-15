import os
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

df=pd.read_csv('data/observations.csv')
df = df[["image_url", "taxon_species_name"]].dropna()
DATASET_DIR = 'bear_dataset'
MAX_WORKERS = 20

print(df["taxon_species_name"].unique())

os.makedirs(DATASET_DIR, exist_ok=True)
species_list = df["taxon_species_name"].unique()
for sp in species_list:
    os.makedirs(os.path.join(DATASET_DIR, sp), exist_ok=True)

failed = []

def download_one(row):
    idx, url, species = row["index"], row["image_url"], row["taxon_species_name"]
    try:
        resp = requests.get(url, timeout=10, stream=True)
        resp.raise_for_status()

        filepath = os.path.join(DATASET_DIR, species, f"{idx}.jpg")
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return None
    except Exception as e:
        return (url, species, str(e)[:100])

rows = df.reset_index()[["index", "image_url", "taxon_species_name"]].to_dict("records")

print(f"Загружаем {len(rows)} изображений")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_one, r) for r in rows]
    for f in tqdm(as_completed(futures), total=len(futures), miniters=100, mininterval=1.0):
        err = f.result()
        if err:
            failed.append(err)

print(f"\n Успешно: {len(rows) - len(failed)} ")
if failed:
    print(f" Ошибок: {len(failed)}")
