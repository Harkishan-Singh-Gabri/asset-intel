import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TARGET_CLASSES = [
    "Laptop",
    "Computer monitor",
    "Computer keyboard",
    "Printer",
    "Telephone",
    "Television"
]

def convert_coco_to_yolo(input_dir="data/raw", output_dir="data/yolo"):
    # Create folder structure YOLO expects
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # Load COCO annotations
    with open(f"{input_dir}/labels.json") as f:
        coco = json.load(f)

    # Filter only our 6 target categories
    coco["categories"] = [
        cat for cat in coco["categories"]
        if cat["name"] in TARGET_CLASSES
    ]
    target_ids = {cat["id"] for cat in coco["categories"]}

    # Filter annotations to only our categories
    coco["annotations"] = [
        ann for ann in coco["annotations"]
        if ann["category_id"] in target_ids
    ]

    # Filter images that have at least one valid annotation
    valid_img_ids = {ann["image_id"] for ann in coco["annotations"]}
    coco["images"] = [
        img for img in coco["images"]
        if img["id"] in valid_img_ids
    ]

    print(f"Categories: {[c['name'] for c in coco['categories']]}")
    print(f"Valid images after filtering: {len(coco['images'])}")
    print(f"Valid annotations after filtering: {len(coco['annotations'])}")

    # Map image_id -> annotations
    img_to_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Map category_id -> 0-indexed class
    cat_id_to_idx = {
        cat["id"]: idx
        for idx, cat in enumerate(coco["categories"])
    }

    # Split 80/20
    images = coco["images"]
    train_imgs, val_imgs = train_test_split(
        images, test_size=0.2, random_state=42
    )

    def process_split(img_list, split_name):
        saved = 0
        for img_info in tqdm(img_list, desc=split_name):
            img_path = f"{input_dir}/data/{img_info['file_name']}"
            if not os.path.exists(img_path):
                continue

            # Copy image
            dst_img = f"{output_dir}/images/{split_name}/{img_info['file_name']}"
            shutil.copy(img_path, dst_img)

            # Convert to YOLO format
            anns = img_to_anns.get(img_info["id"], [])
            W, H = img_info["width"], img_info["height"]

            yolo_lines = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                cx = (x + w/2) / W
                cy = (y + h/2) / H
                nw = w / W
                nh = h / H
                cls = cat_id_to_idx[ann["category_id"]]
                yolo_lines.append(
                    f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

            # Save label file
            label_name = Path(img_info["file_name"]).stem + ".txt"
            with open(f"{output_dir}/labels/{split_name}/{label_name}", "w") as f:
                f.write("\n".join(yolo_lines))
            saved += 1

        print(f"{split_name}: {saved} images saved")

    process_split(train_imgs, "train")
    process_split(val_imgs, "val")

    # Create dataset.yaml with absolute path
    abs_path = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val
nc: {len(coco['categories'])}
names: {[cat['name'] for cat in coco['categories']]}
"""
    with open(f"{output_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDone")
    print(f"dataset.yaml -> {output_dir}/dataset.yaml")
    print(f"Path set to: {abs_path}")

if __name__ == "__main__":
    convert_coco_to_yolo()