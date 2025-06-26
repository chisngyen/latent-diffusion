from datasets import load_dataset, load_from_disk
import os
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

def download_dataset(output_path="../../data"):
    """
    Tải dataset từ Hugging Face về local
    """
    print("🔄 Đang tải dataset từ Hugging Face...")
    
    # Tạo thư mục data
    os.makedirs(output_path, exist_ok=True)
    
    # Tải dataset
    dataset = load_dataset("Brvcket/sdxl-inpainting-lights")
    
    # Lưu dataset xuống local
    dataset_path = os.path.join(output_path, "sdxl-inpainting-lights")
    dataset.save_to_disk(dataset_path)
    
    print(f"✅ Dataset đã được tải về: {dataset_path}")
    print(f"📊 Dataset info: {dataset}")
    
    if 'train' in dataset:
        print(f"📈 Train samples: {len(dataset['train'])}")
        print(f"🔤 Columns: {dataset['train'].column_names}")
        
        # Xem sample đầu tiên
        sample = dataset['train'][0]
        print(f"🔍 Sample structure: {list(sample.keys())}")
    
    return dataset_path

if __name__ == "__main__":
    download_dataset()