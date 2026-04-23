import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm

#Warehouse conditions transforms
WAREHOUSE_TRANSFORM = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10,50), p=0.4),
    A.Perspective(scale=(0.05,0.1), p=0.3),
    A.HueSaturationValue(p=0.3),
    A.Resize(640,640)
], bbox_params=A.BboxParams(
    format='coco',
    label_fields=['category_ids'],
    min_visibility=0.3
)
)

def augment_dataset(input_dir="data/raw",output_dir="data/processed",augment_factor=2):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    #load COCO annotations
    ann_file=f"{input_dir}/labels.json"
    if not os.path.exists(ann_file):
        print(f"Annotations not found at {ann_file}")
        return
    
    with open(ann_file) as f:
        coco=json.load(f)

    #map the image_id to annotations
    img_to_anns={}
    for ann in coco['annotations']:
        img_id=ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id]=[]
        img_to_anns[img_id].append(ann)

    print(f"Augmenting {len(coco['images'])} images x{augment_factor}...")

    for img_info in tqdm(coco['images']):
        img_path=f"{input_dir}/data/{img_info['file_name']}"
        if not os.path.exists(img_path):
            continue

        image=cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        anns=img_to_anns.get(img_info['id'],[])
        bboxes=[a['bbox'] for a in anns]
        category_ids=[a['category_id'] for a in anns]

        for i in range(augment_factor):
            try:
                augmented=WAREHOUSE_TRANSFORM(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                out_name=f"{Path(img_info['file_name']).stem}_aug{i}.jpg"
                out_img=cv2.cvtColor(augmented['image'],cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{output_dir}/images/{out_name}",out_img)
            except Exception as e:
                continue
    print("Augmentation completed.")

if __name__=="__main__":
    augment_dataset()