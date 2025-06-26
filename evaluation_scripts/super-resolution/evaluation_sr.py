# evaluate_urban100.py
import argparse
import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# Các import từ code gốc
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# Import cho FID score
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    FID_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
    FID_AVAILABLE = False

# Import cho LPIPS (Visual Quality)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False

# Import cho SSIM
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_model_from_config(config, ckpt):
    """Load model từ config và checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def preprocess_image(image_path, target_size=256):
    """Preprocess image để có shape đúng [1, 3, H, W]"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)
    
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Add batch dim
    image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    
    return image_tensor

def super_resolve_single_image(model, lr_image_path, target_size=256):
    """Super resolve một ảnh đơn lẻ"""
    # Load và preprocess LR image
    lr_tensor = preprocess_image(lr_image_path, target_size=target_size//4).cuda()
    
    # Upsample LR để tạo dummy HR cho shape reference
    hr_dummy = torch.nn.functional.interpolate(
        lr_tensor, 
        scale_factor=4.0,
        mode='bicubic', 
        align_corners=False
    )
    
    with torch.no_grad():
        with model.ema_scope():
            # Get latent encoding của dummy HR (để biết target shape)
            z_hr = model.encode_first_stage(hr_dummy)
            
            # Resize LR image về spatial size của latent
            latent_size = z_hr.shape[-1]
            lr_resized = torch.nn.functional.interpolate(
                lr_tensor,
                size=(latent_size, latent_size),
                mode='bicubic',
                align_corners=False
            )
            
            # Tạo noise tensor
            noise = torch.randn_like(z_hr)
            
            # Super resolution với DDIM
            sampler = DDIMSampler(model)
            conditioning = lr_resized
            
            samples, _ = sampler.sample(
                S=50,
                conditioning=conditioning,
                batch_size=1,
                shape=z_hr.shape[1:],
                verbose=False,
                eta=0.0,
                x_T=noise
            )
            
            # Decode samples
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            
            return x_samples

def calculate_fid_features(images, inception_model):
    """Tính features cho FID score"""
    features = []
    batch_size = 8
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack([
            torch.from_numpy(np.array(img).transpose(2, 0, 1) / 255.0).float()
            for img in batch
        ]).cuda()
        
        # Resize to 299x299 for Inception
        if batch_tensor.shape[-1] != 299:
            batch_tensor = F.interpolate(batch_tensor, size=(299, 299), mode='bilinear')
        
        with torch.no_grad():
            pred = inception_model(batch_tensor)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            features.append(pred.cpu().numpy().reshape(pred.size(0), -1))
    
    return np.concatenate(features, axis=0)

def calculate_fid_score(real_images, generated_images):
    """Tính FID score giữa real và generated images"""
    if not FID_AVAILABLE:
        return None
    
    try:
        # Khởi tạo Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx]).cuda()
        inception_model.eval()
        
        # Tính features
        real_features = calculate_fid_features(real_images, inception_model)
        generated_features = calculate_fid_features(generated_images, inception_model)
        
        # Tính FID
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value
        
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None

def calculate_precision_recall(real_images, generated_images, k=3):
    """Tính Precision và Recall scores"""
    def compute_distances(X, Y):
        """Tính distance matrix giữa hai sets of images"""
        X_flat = X.reshape(X.shape[0], -1)
        Y_flat = Y.reshape(Y.shape[0], -1)
        
        # Compute pairwise L2 distances
        distances = torch.cdist(
            torch.from_numpy(X_flat).float(),
            torch.from_numpy(Y_flat).float()
        ).numpy()
        
        return distances
    
    # Convert images to numpy arrays
    real_arrays = np.array([np.array(img) for img in real_images])
    generated_arrays = np.array([np.array(img) for img in generated_images])
    
    # Compute distances
    real_to_gen = compute_distances(real_arrays, generated_arrays)
    gen_to_real = compute_distances(generated_arrays, real_arrays)
    
    # Precision: For each generated image, check if its k-NN in real images
    # are close enough
    precision_scores = []
    for i in range(len(generated_images)):
        distances_to_real = real_to_gen[:, i]
        knn_distances = np.partition(distances_to_real, k)[:k]
        precision_scores.append(np.mean(knn_distances))
    
    # Recall: For each real image, check if its k-NN in generated images
    # are close enough  
    recall_scores = []
    for i in range(len(real_images)):
        distances_to_gen = gen_to_real[:, i]
        knn_distances = np.partition(distances_to_gen, k)[:k]
        recall_scores.append(np.mean(knn_distances))
    
    # Lower distances = higher precision/recall
    precision = 1.0 / (1.0 + np.mean(precision_scores))
    recall = 1.0 / (1.0 + np.mean(recall_scores))
    
    return precision, recall

def calculate_visual_quality(real_images, generated_images):
    """Tính Visual Quality metrics (LPIPS, SSIM, PSNR)"""
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    
    # Initialize LPIPS if available
    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net='alex').cuda()
    
    for real_img, gen_img in zip(real_images, generated_images):
        # Convert to numpy arrays
        real_np = np.array(real_img)
        gen_np = np.array(gen_img)
        
        # SSIM
        ssim_val = ssim(real_np, gen_np, multichannel=True, channel_axis=2, data_range=255)
        ssim_scores.append(ssim_val)
        
        # PSNR
        psnr_val = psnr(real_np, gen_np, data_range=255)
        psnr_scores.append(psnr_val)
        
        # LPIPS
        if lpips_model is not None:
            real_tensor = torch.from_numpy(real_np.transpose(2, 0, 1) / 255.0 * 2 - 1).float().unsqueeze(0).cuda()
            gen_tensor = torch.from_numpy(gen_np.transpose(2, 0, 1) / 255.0 * 2 - 1).float().unsqueeze(0).cuda()
            
            with torch.no_grad():
                lpips_val = lpips_model(real_tensor, gen_tensor).item()
                lpips_scores.append(lpips_val)
    
    results = {
        'ssim': np.mean(ssim_scores),
        'psnr': np.mean(psnr_scores),
    }
    
    if lpips_scores:
        results['lpips'] = np.mean(lpips_scores)
    
    return results

def calculate_edge_consistency(real_images, generated_images):
    """Tính Edge Consistency score"""
    edge_scores = []
    
    for real_img, gen_img in zip(real_images, generated_images):
        # Convert to grayscale
        real_gray = cv2.cvtColor(np.array(real_img), cv2.COLOR_RGB2GRAY)
        gen_gray = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny
        real_edges = cv2.Canny(real_gray, 100, 200)
        gen_edges = cv2.Canny(gen_gray, 100, 200)
        
        # Calculate edge consistency as Intersection over Union
        intersection = np.logical_and(real_edges, gen_edges).sum()
        union = np.logical_or(real_edges, gen_edges).sum()
        
        if union > 0:
            edge_consistency = intersection / union
        else:
            edge_consistency = 1.0
            
        edge_scores.append(edge_consistency)
    
    return np.mean(edge_scores)

def evaluate_urban100(dataset_path, model_config, model_checkpoint, scale_factor=4, output_dir="evaluation_results"):
    """Evaluate trên dataset Urban100"""
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    config = OmegaConf.load(model_config)
    model = load_model_from_config(config, model_checkpoint)
    
    # Paths
    scale_dir = f"X{scale_factor}"
    hr_dir = os.path.join(dataset_path, scale_dir, "HIGH")
    lr_dir = os.path.join(dataset_path, scale_dir, "LOW")
    
    if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
        print(f"Error: Directory {hr_dir} or {lr_dir} not found!")
        return
    
    # Get image lists
    hr_images = sorted(glob.glob(os.path.join(hr_dir, f"*_SRF_{scale_factor}_HR.*")))
    lr_images = sorted(glob.glob(os.path.join(lr_dir, f"*_SRF_{scale_factor}_LR.*")))
    
    print(f"Found {len(hr_images)} HR images and {len(lr_images)} LR images")
    
    if len(hr_images) != len(lr_images):
        print("Warning: Number of HR and LR images don't match!")
    
    # Lists to store images for evaluation
    real_images = []
    generated_images = []
    
    print("Generating super-resolved images...")
    for i, (hr_path, lr_path) in enumerate(tqdm(zip(hr_images, lr_images), total=len(hr_images))):
        try:
            # Load ground truth HR image
            hr_img = Image.open(hr_path).convert("RGB")
            hr_img = hr_img.resize((256, 256), Image.BICUBIC)
            real_images.append(hr_img)
            
            # Generate super-resolved image
            sr_tensor = super_resolve_single_image(model, lr_path, target_size=256)
            sr_img_np = sr_tensor[0].cpu().permute(1, 2, 0).numpy()
            sr_img_np = (sr_img_np * 255).astype(np.uint8)
            sr_img = Image.fromarray(sr_img_np)
            generated_images.append(sr_img)
            
            # Save result (optional, for visual inspection)
            if i < 10:  # Save first 10 results
                result_dir = os.path.join(output_dir, "sample_results")
                os.makedirs(result_dir, exist_ok=True)
                sr_img.save(os.path.join(result_dir, f"result_{i:03d}_sr.png"))
                hr_img.save(os.path.join(result_dir, f"result_{i:03d}_hr.png"))
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    print(f"Successfully processed {len(generated_images)} images")
    
    # Calculate metrics
    results = {}
    
    print("Calculating FID Score...")
    fid_score = calculate_fid_score(real_images, generated_images)
    if fid_score is not None:
        results['fid_score'] = fid_score
        print(f"FID Score: {fid_score:.4f}")
    
    print("Calculating Precision and Recall...")
    precision, recall = calculate_precision_recall(real_images, generated_images)
    results['precision'] = precision
    results['recall'] = recall
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("Calculating Visual Quality metrics...")
    visual_quality = calculate_visual_quality(real_images, generated_images)
    results['visual_quality'] = visual_quality
    print(f"Visual Quality - SSIM: {visual_quality['ssim']:.4f}")
    print(f"Visual Quality - PSNR: {visual_quality['psnr']:.4f}")
    if 'lpips' in visual_quality:
        print(f"Visual Quality - LPIPS: {visual_quality['lpips']:.4f}")
    
    print("Calculating Edge Consistency...")
    edge_consistency = calculate_edge_consistency(real_images, generated_images)
    results['edge_consistency'] = edge_consistency
    print(f"Edge Consistency: {edge_consistency:.4f}")
    
    # Save results
    results['dataset'] = 'Urban100'
    results['scale_factor'] = scale_factor
    results['num_images'] = len(generated_images)
    results['timestamp'] = datetime.now().isoformat()
    
    results_file = os.path.join(output_dir, f"urban100_x{scale_factor}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset: Urban100 (X{scale_factor})")
    print(f"Number of images: {len(generated_images)}")
    if fid_score is not None:
        print(f"FID Score: {fid_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"SSIM: {visual_quality['ssim']:.4f}")
    print(f"PSNR: {visual_quality['psnr']:.4f}")
    if 'lpips' in visual_quality:
        print(f"LPIPS: {visual_quality['lpips']:.4f}")
    print(f"Edge Consistency: {edge_consistency:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Super Resolution on Urban100 dataset')
    parser.add_argument('--dataset_path', type=str, default="data/super-resolution_dataset",
                       help='Path to Urban100 dataset')
    parser.add_argument('--config', type=str, default='models/ldm/bsr_sr/config.yaml',
                       help='Path to model config')
    parser.add_argument('--checkpoint', type=str, default='models/ldm/bsr_sr/model.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                       help='Scale factor (2 or 4)')
    parser.add_argument('--output_dir', type=str, default='evaluation_sr',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_urban100(
        dataset_path=args.dataset_path,
        model_config=args.config,
        model_checkpoint=args.checkpoint,
        scale_factor=args.scale,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()