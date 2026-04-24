import fiftyone.zoo as foz
import fiftyone as fo

CATEGORIES=[
    'Laptop',
    'Computer monitor',
    'Printer',
    'Computer keyboard',
    'Telephone',
    'Television'
]

def download_hardware_images():
    print("Downloading Open Images V7 hardware subset...")
    dataset=foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=CATEGORIES,
        max_samples=1000,
        dataset_name="hardare-assets"
    )

    print(f"Downloaded {len(dataset)} images")

    #Export
    dataset.export(
        export_dir="data/raw",
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth"
    )

    print("Exported to raw/data successfully")
    print(f"Total images exported: {len(dataset)}")

if __name__=="__main__":
    download_hardware_images()