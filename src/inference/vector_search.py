import faiss
import open_clip
import torch
import chromadb
import numpy as np
from PIL import Image

FAISS_INDEX="data/embeddings/assets.faiss"
CHROMA_DIR="data/embeddings/chroma"

print("Loading CLIP + index...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()
index = faiss.read_index(FAISS_INDEX)
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("assets")
print("Ready")

def encode_image(img: Image.Image)->np.ndarray:
    tensor=preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec=model.encode_image(tensor)
        vec=vec/vec.norm()
    return vec.numpy().astype('float32')

def match_asset(img:Image.Image, top_k=0.5):
    #encode query image
    q_vec=encode_image(img)

    #FAISS search
    distances,indices=index.search(q_vec,top_k)

    #get metadata from ChromaDB
    results=[]
    for dist,idx in zip(distances[0],indices[0]):
        try:
            res=collection.get(ids=[str(idx)])
            if res["metadatas"]:
                results.append({
                    "metadata": res['metadatas'][0],
                    "score": float(dist)
                })
        except:
            continue
    return results