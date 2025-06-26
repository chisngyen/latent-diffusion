# Latent Diffusion Models

[arXiv](https://arxiv.org/abs/2112.10752) | [BibTeX](#bibtex)

<p align="center">
<img src=assets/results.gif />
</p>

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>

<p align="center">
<img src=assets/modelfigure.png />
</p>

## Requirements

```
conda env create -f environment.yaml
conda activate ldm
```

# Đánh giá Latent Diffusion Models

Đánh giá toàn diện cho các mô hình Latent Diffusion qua ba tác vụ chính: **Inpainting**, **Super Resolution**, và **Text-to-Image Generation**.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Thiết Lập Dataset](#thiết-lập-dataset)
- [Cách Sử Dụng](#cách-sử-dụng)
- [Metrics Đánh Giá](#metrics-đánh-giá)
- [Cấu Trúc Output](#cấu-trúc-output)

## 🎯 Tổng Quan

Bộ công cụ đánh giá này cung cấp benchmark chuẩn hóa cho ba nhiệm vụ chính của mô hình diffusion:

### 1. **Đánh Giá Inpainting**

- **Dataset**: SDXL Inpainting Lights
- **Metrics**: FID, LPIPS, Edge Consistency, SSIM, PSNR
- **Mục đích**: Đánh giá chất lượng hoàn thiện ảnh và độ mượt mà của ranh giới

### 2. **Đánh Giá Super Resolution**

- **Dataset**: Urban100
- **Metrics**: FID, Precision/Recall, Visual Quality, Edge Consistency
- **Mục đích**: Đánh giá chất lượng upscaling và bảo toàn chi tiết

### 3. **Đánh Giá Text-to-Image**

- **Dataset**: Flickr30k
- **Metrics**: FID, Precision/Recall, CLIP Score
- **Mục đích**: Đo lường sự phù hợp giữa text-image và chất lượng sinh ảnh

## 🛠 Yêu Cầu Hệ Thống

### Dependencies Cơ Bản

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pillow>=8.0.0
opencv-python>=4.5.0
scikit-image>=0.18.0
scipy>=1.7.0
tqdm>=4.62.0
omegaconf>=2.1.0
einops>=0.3.0
```

### Dependencies Tùy Chọn

```
lpips>=0.1.4           # Cho perceptual similarity
clip-by-openai>=1.0    # Cho CLIP scores  
pytorch-fid>=0.3.0     # Cho tính toán FID
datasets>=2.0.0        # Cho xử lý dataset
matplotlib>=3.5.0      # Cho visualization
```

### Dependencies Mô Hình

```
# Các component của mô hình latent diffusion
ldm/                   # Modules mô hình local
main.py               # Utilities khởi tạo mô hình
```

## 📦 Cài Đặt

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd latent-diffusion
```

### 2. Cài Đặt Dependencies

```bash
# Cài đặt cơ bản
pip install torch torchvision torchaudio
pip install numpy pillow opencv-python scikit-image scipy tqdm omegaconf einops

# Tùy chọn nhưng được khuyến nghị
pip install lpips clip-by-openai pytorch-fid datasets matplotlib

# Cho xử lý dataset
pip install datasets
```

### 3. Kiểm Tra Cài Đặt

```bash
python -c "import torch, torchvision, cv2, lpips, clip; print('✅ Tất cả dependencies đã được cài đặt')"
```

## 💾 Thiết Lập Dataset

### 📝 1. Dataset Inpainting

#### Thiết Lập Cấu Trúc Dữ Liệu

```bash
cd evaluation_scripts/inpainting/
python load_data.py
python setup_data.py
```

**Cấu Trúc Mong Đợi:**

```
data/
├── sdxl-inpainting-lights/          # Dataset gốc
└── inpainting_dataset/              # Dataset đã xử lý
    ├── images/                      # Ảnh gốc (PNG)
    ├── masked_images/              # Ảnh đã mask (PNG)  
    ├── masks/                      # Binary masks (PNG)
    ├── texts/                      # Text prompts (TXT)
    ├── metadata.json               # Metadata dataset
    └── stats.json                  # Thống kê dataset
```

### 🔍 2. Dataset Super Resolution

#### Tải dataset

```bash
https://www.kaggle.com/datasets/harshraone/urban100
```

**Cấu Trúc Mong Đợi:**

```
data/super-resolution_dataset/
├── X2/                             # Scale factor 2
│   ├── HIGH/                       # Ảnh HR
│   │   └── *_SRF_2_HR.*
│   └── LOW/                        # Ảnh LR  
│       └── *_SRF_2_LR.*
└── X4/                             # Scale factor 4
    ├── HIGH/                       # Ảnh HR
    │   └── *_SRF_4_HR.*
    └── LOW/                        # Ảnh LR
        └── *_SRF_4_LR.*
```

### 📸 3. Dataset Text-to-Image

#### Tải dataset

```bash
https://www.kaggle.com/datasets/adityajn105/flickr30k
```

**Cấu Trúc Mong Đợi:**

```
data/text2img_dataset/
├── images/                         # Files ảnh
│   ├── 1000092795.jpg
│   ├── 1000092795.jpg
│   └── ...
└── captions.txt                    # CSV với image_name,caption
```

**Format captions.txt:**

```csv
image_name,caption
1000092795.jpg,Two young girls riding their scooters on a sidewalk.
1001773457.jpg,A brown dog is running in a field with tall grass.
...

## 🚀 Cách Sử Dụng
### Lưu ý: Giải nén các file models.zip của các models trước khi sử dụng (models/first_stage_models + models/ldm)

### 🎨 1. Đánh Giá Inpainting

#### Cách Sử Dụng Cơ Bản
```bash
cd evaluation_scripts/inpainting/
python evaluation_inpainting.py \
    --data_dir ../../data/inpainting_dataset \
    --output_dir ../../evaluation/inpainting_results \
    --config ../../models/ldm/inpainting_big/config.yaml \
    --model ../../models/ldm/inpainting_big/last.ckpt \
    --max_samples 100 \
    --steps 50
```

#### Tùy Chọn Nâng Cao

```bash
python evaluation_inpainting.py \
    --data_dir ../../data/inpainting_dataset \
    --output_dir ../../evaluation/inpainting_full \
    --config ../../models/ldm/inpainting_big/config.yaml \
    --model ../../models/ldm/inpainting_big/last.ckpt \
    --max_samples 1000 \          # Nhiều samples hơn cho đánh giá toàn diện
    --steps 100                   # Nhiều steps hơn cho chất lượng tốt hơn
```

### 🔍 2. Đánh Giá Super Resolution

#### Scale Factor 2

```bash
cd evaluation_scripts/super-resolution/
python evaluation_sr.py \
    --dataset_path ../../data/super-resolution_dataset \
    --config ../../models/ldm/bsr_sr/config.yaml \
    --checkpoint ../../models/ldm/bsr_sr/model.ckpt \
    --scale 2 \
    --output_dir ../../evaluation/sr_x2_results
```

#### Scale Factor 4

```bash
python evaluation_sr.py \
    --dataset_path ../../data/super-resolution_dataset \
    --config ../../models/ldm/bsr_sr/config.yaml \
    --checkpoint ../../models/ldm/bsr_sr/model.ckpt \
    --scale 4 \
    --output_dir ../../evaluation/sr_x4_results
```

### 📝 3. Đánh Giá Text-to-Image

#### Test Nhanh (Lưu 10% ảnh)

```bash
cd evaluation_scripts/text2img/
python evaluation_text2img.py \
    --dataset_root ../../data/text2img_dataset \
    --config_path ../../configs/latent-diffusion/txt2img-1p4B-eval.yaml \
    --model_path ../../models/ldm/text2img-large/model.ckpt \
    --output_dir ../../evaluation/text2img_test \
    --max_samples 100 \
    --save_images_ratio 0.1 \
    --ddim_steps 50
```

#### Đánh Giá Đầy Đủ

```bash
python evaluation_text2img.py \
    --dataset_root ../../data/text2img_dataset \
    --config_path ../../configs/latent-diffusion/txt2img-1p4B-eval.yaml \
    --model_path ../../models/ldm/text2img-large/model.ckpt \
    --output_dir ../../evaluation/text2img_full \
    --max_samples 1000 \
    --save_images_ratio 1.0 \     # Lưu tất cả ảnh được sinh
    --ddim_steps 100 \
    --scale 7.5                   # Classifier-free guidance scale
```

## 📊 Metrics Đánh Giá

### 🎨 Metrics Inpainting


| Metric               | Mô Tả                                                       | Khoảng | Tốt Hơn     |
| ---------------------- | --------------------------------------------------------------- | --------- | --------------- |
| **FID Score**        | Fréchet Inception Distance - Độ thực của ảnh tổng thể | [0, ∞) | ↓ Thấp hơn |
| **LPIPS**            | Learned Perceptual Image Patch Similarity                     | [0, 1]  | ↓ Thấp hơn |
| **Edge Consistency** | Độ mượt mà ranh giới tại viền mask                    | [0, 1]  | ↑ Cao hơn   |
| **SSIM**             | Structural Similarity Index                                   | [0, 1]  | ↑ Cao hơn   |
| **PSNR**             | Peak Signal-to-Noise Ratio                                    | [0, ∞) | ↑ Cao hơn   |

**Cách Đánh Giá Chất Lượng:**

- **FID < 30**: 🌟 Xuất sắc
- **FID < 50**: ✅ Rất tốt
- **FID < 100**: 👍 Tốt
- **LPIPS < 0.1**: 🌟 Chất lượng perceptual xuất sắc
- **Edge Consistency > 0.8**: 🌟 Pha trộn xuất sắc

### 🔍 Metrics Super Resolution


| Metric        | Mô Tả                              | Khoảng | Tốt Hơn     |
| --------------- | -------------------------------------- | --------- | --------------- |
| **FID Score** | Độ tương đồng phân phối ảnh | [0, ∞) | ↓ Thấp hơn |
| **Precision** | Chất lượng sample được sinh    | [0, 1]  | ↑ Cao hơn   |
| **Recall**    | Độ bao phủ phân phối thực      | [0, 1]  | ↑ Cao hơn   |
| **SSIM**      | Bảo toàn cấu trúc                | [0, 1]  | ↑ Cao hơn   |
| **PSNR**      | Chất lượng giảm nhiễu           | [0, ∞) | ↑ Cao hơn   |
| **LPIPS**     | Độ tương đồng perceptual       | [0, 1]  | ↓ Thấp hơn |

### 📝 Metrics Text-to-Image


| Metric         | Mô Tả                             | Khoảng | Tốt Hơn     |
| ---------------- | ------------------------------------- | --------- | --------------- |
| **FID Score**  | Phân phối ảnh sinh vs ảnh thực | [0, ∞) | ↓ Thấp hơn |
| **CLIP Score** | Độ phù hợp semantic text-image  | [-1, 1] | ↑ Cao hơn   |
| **Precision**  | Chất lượng sample được sinh   | [0, 1]  | ↑ Cao hơn   |
| **Recall**     | Độ bao phủ phân phối thực     | [0, 1]  | ↑ Cao hơn   |

**Cách Đánh Giá CLIP Score:**

- **> 0.3**: 🌟 Độ phù hợp text-image xuất sắc
- **> 0.25**: ✅ Độ phù hợp tốt
- **> 0.2**: 👍 Độ phù hợp khá

## 📁 Cấu Trúc Output

### Kết Quả Inpainting

```
evaluation/inpainting_results/
├── evaluation_results.json         # Metrics đầy đủ và logs
├── inpainted_00001.png             # Kết quả mẫu
├── inpainted_00002.png
└── ...
```

### Kết Quả Super Resolution

```
evaluation/sr_x4_results/
├── urban100_x4_results.json        # Metrics đánh giá
├── sample_results/                 # 10 kết quả đầu tiên
│   ├── result_000_sr.png           # Super-resolved
│   ├── result_000_hr.png           # Ground truth
│   └── ...
```

### Kết Quả Text-to-Image

```
evaluation/text2img_full/
├── evaluation_results.txt          # Kết quả chi tiết
├── generated_00001.png             # Ảnh được sinh
├── generated_00002.png
└── ...
```

# BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```
