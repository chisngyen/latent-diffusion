from datasets import load_from_disk
import os
from PIL import Image
import json
from tqdm import tqdm

def extract_dataset_images(dataset_path, output_dir):
    """
    Extract tất cả ảnh và text từ dataset ra thư mục
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    train_data = dataset['train']
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo các thư mục con
    folders = {
        'images': os.path.join(output_dir, 'images'),           # Ảnh gốc
        'masked_images': os.path.join(output_dir, 'masked_images'), # Ảnh đã mask
        'masks': os.path.join(output_dir, 'masks'),             # Mask
        'texts': os.path.join(output_dir, 'texts')              # Text prompts
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Metadata để track mapping
    metadata = []
    
    print(f"Extracting {len(train_data)} samples...")
    
    # Extract từng sample
    for i, sample in enumerate(tqdm(train_data, desc="Extracting")):
        # Tạo filename với zero-padding
        filename = f"{i:05d}"
        
        # Extract ảnh gốc
        if 'image' in sample and sample['image'] is not None:
            image_path = os.path.join(folders['images'], f"{filename}.png")
            sample['image'].save(image_path)
        
        # Extract ảnh đã mask
        if 'masked_image' in sample and sample['masked_image'] is not None:
            masked_path = os.path.join(folders['masked_images'], f"{filename}.png")
            sample['masked_image'].save(masked_path)
        
        # Extract mask
        if 'mask' in sample and sample['mask'] is not None:
            mask_path = os.path.join(folders['masks'], f"{filename}.png")
            sample['mask'].save(mask_path)
        
        # Extract text
        if 'text' in sample and sample['text'] is not None:
            text_path = os.path.join(folders['texts'], f"{filename}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(sample['text'])
        
        # Lưu metadata
        metadata.append({
            'id': i,
            'filename': filename,
            'text': sample.get('text', ''),
            'image_size': list(sample['image'].size) if 'image' in sample else None
        })
    
    # Lưu metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Lưu thống kê
    stats = {
        'total_samples': len(train_data),
        'folders_created': list(folders.keys()),
        'sample_structure': list(train_data[0].keys()) if len(train_data) > 0 else []
    }
    
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Extraction hoàn thành!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total samples: {len(train_data)}")
    print(f"📁 Folders created:")
    for name, path in folders.items():
        count = len([f for f in os.listdir(path) if f.endswith(('.png', '.txt'))])
        print(f"   - {name}: {count} files")
    print(f"📄 Metadata saved: {metadata_path}")
    print(f"📈 Stats saved: {stats_path}")

def preview_extracted_data(output_dir, num_samples=3):
    """
    Preview một vài sample đã extract
    """
    import matplotlib.pyplot as plt
    
    # Load metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print("Metadata file not found!")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Preview
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(metadata))):
        filename = metadata[i]['filename']
        
        # Load và hiển thị ảnh
        folders = ['images', 'masked_images', 'masks']
        titles = ['Original', 'Masked', 'Mask', 'Text']
        
        for j, folder in enumerate(folders):
            img_path = os.path.join(output_dir, folder, f"{filename}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if folder == 'masks':
                    axes[i, j].imshow(img, cmap='gray')
                else:
                    axes[i, j].imshow(img)
            axes[i, j].set_title(titles[j])
            axes[i, j].axis('off')
        
        # Hiển thị text
        text_path = os.path.join(output_dir, 'texts', f"{filename}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            axes[i, 3].text(0.1, 0.5, text, transform=axes[i, 3].transAxes,
                           fontsize=10, wrap=True, va='center')
        axes[i, 3].set_title('Text')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Cấu hình
    dataset_path = "./data/sdxl-inpainting-lights"
    output_dir = "./data/inpainting_dataset"
    
    # Extract
    extract_dataset_images(dataset_path, output_dir)
    
    # Preview
    print("\n" + "="*50)
    print("Preview extracted data:")
    preview_extracted_data(output_dir, num_samples=3)