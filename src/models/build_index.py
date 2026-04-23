import open_clip
import torch
import faiss
import numpy as np
import chromadb
import json
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

#Paths
RAW_DIR="data/raw"
EMBED_DIR="data/embeddings"
FAISS_INDEX=f"{EMBED_DIR}/assets.faiss"
CHROMA_DIR=f"{EMBED_DIR}/chroma"

def load_clip():
    print("Loading CLIP model...")
    model,_,preprocess=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k")
    model.eval()
    print("Clip loaded")
    return model,preprocess

def encode_image(model,preprocess,img_path):
    img=preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding=model.encode_image(img)
        embedding=embedding/embedding.norm()  #L2 normalize
    return embedding.numpy().astype("float32")

def build_index():
    os.makedirs(EMBED_DIR,exist_ok=True)
    
    #load CLIP model
    model,preprocess=load_clip()

    #load asset catalog from COCO labels
    with open(f"{RAW_DIR}/labels.json") as f:
        coco=json.load(f)

    #build image list
    images=coco['images']
    categories={cat['id']:cat['name'] for cat in coco['categories']}

    #map image_id to category
    img_to_cat={}
    for ann in coco['annotations']:
        img_to_cat[ann['image_id']]=categories[ann['category_id']]
    print(f"Building index for {len(images)} images...")

    #FAISS index
    dim=512
    index=faiss.IndexFlatIP(dim)

    #ChromaDB
    chroma=chromadb.PersistentClient(path=CHROMA_DIR)
    collection=chroma.get_or_create_collection("assets")

    ids,embeddings,metadatas=[],[],[]
    for img_info in tqdm(images):
        img_path=f"{RAW_DIR}/data/{img_info['file_name']}"
        if not os.path.exists(img_path):
            continue

        try:
            vec=encode_image(model,preprocess,img_path)
            index.add(vec)
            asset_id=str(img_info['id'])
            category=img_to_cat.get(img_info['id'],'unknown')

            ids.append(asset_id)
            embeddings.append(vec[0].tolist())
            metadatas.append({
                "image_file":img_info['file_name'],
                "category": category,
                "asset_id": asset_id
            })
        except Exception as e:
            continue

    #save to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    #save FAISS index
    faiss.write_index(index,FAISS_INDEX)

    print(f"\nIndex built")
    print(f"Total vectors: {index.ntotal}")
    print(f"FAISS index -> {FAISS_INDEX}")
    print(f"ChromaDB -> {CHROMA_DIR}")

if __name__ == "__main__":
    build_index()
