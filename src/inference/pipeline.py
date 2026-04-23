from ultralytics import YOLO
from PIL import Image
from .vector_search import match_asset
import os

#load YOLO
YOLO_WEIGHTS = "models/yolo/v1/weights/best.pt"
yolo = YOLO(YOLO_WEIGHTS)
print(f"YOLO loaded")

def process_photo(image_path:str):
    img=Image.open(image_path).convert("RGB")
    results=yolo(image_path, verbose=False)
    catalogued=[]

    for r in results:
        if len(r.boxes)==0:
            print("No items detected in photo")
            continue

        for box in r.boxes:
            x1,y1,x2,y2=[int(v) for v in box.xyxy[0]]
            crop=img.crop((x1,y1,x2,y2))

            #CLIP vector search
            matches=match_asset(crop,top_k=5)

            catalogued.append({
                "detected_class": r.names[int(box.cls)],
                "yolo_confidence": round(float(box.conf),2),
                "top_match": matches[0] if matches else None,
                "alternatives": matches[1:] if matches else [],
                "bbox": [x1,y1,x2,y2]
            })
    return catalogued