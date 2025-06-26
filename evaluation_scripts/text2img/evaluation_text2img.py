import argparse
import os
import sys
import glob
import csv
import torch
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from einops import rearrange
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing text-to-image generation components
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Load captions
        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    image_name = row[0].strip()
                    caption = row[1].strip()
                    image_path = os.path.join(root_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            'image_path': image_path,
                            'caption': caption,
                            'image_name': image_name
                        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'caption': item['caption'],
            'image_name': item['image_name']
        }


class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained InceptionV3
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = torch.nn.Identity()  # Remove final layer
        self.inception = self.inception.to(device)
        self.inception.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_features(self, images):
        """Extract features from images using InceptionV3"""
        features_list = []
        
        with torch.no_grad():
            for img in tqdm(images, desc="Extracting features"):
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, torch.Tensor):
                    # Convert tensor to PIL Image
                    img = transforms.ToPILImage()(img)
                
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                features = self.inception(img_tensor)
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_fid(self, real_features, generated_features):
        """Calculate FID score between real and generated features"""
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid_score


class PrecisionRecallCalculator:
    def __init__(self, k=3):
        self.k = k
    
    def compute_precision_recall(self, real_features, generated_features):
        """Compute precision and recall using k-nearest neighbors"""
        # Compute distances
        real_distances = self._compute_knn_distances(real_features, real_features)
        gen_distances = self._compute_knn_distances(generated_features, real_features)
        
        # Compute precision: fraction of generated samples that have their 
        # k-nearest neighbor in real samples closer than k-th nearest neighbor in real samples
        precision = np.mean(gen_distances < real_distances)
        
        # Compute recall: fraction of real samples that have their 
        # k-nearest neighbor in generated samples closer than k-th nearest neighbor in real samples
        real_to_gen_distances = self._compute_knn_distances(real_features, generated_features)
        recall = np.mean(real_to_gen_distances < real_distances)
        
        return precision, recall
    
    def _compute_knn_distances(self, X, Y):
        """Compute k-th nearest neighbor distances"""
        distances = []
        for x in X:
            dists = np.linalg.norm(Y - x, axis=1)
            dists = np.sort(dists)
            distances.append(dists[self.k] if len(dists) > self.k else dists[-1])
        return np.array(distances)


class CLIPScoreCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
    def compute_clip_score(self, images, captions):
        """Compute CLIP score between images and captions"""
        scores = []
        
        with torch.no_grad():
            for img, caption in tqdm(zip(images, captions), desc="Computing CLIP scores"):
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                
                image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                text_tensor = clip.tokenize([caption]).to(self.device)
                
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                score = torch.cosine_similarity(image_features, text_features).item()
                scores.append(score)
        
        return np.array(scores)


def load_model_from_config(config, ckpt, verbose=False):
    """Load text-to-image model"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def generate_images_for_evaluation(model, sampler, captions, opt, output_dir, save_images_ratio=1.0):
    """Generate images from captions for evaluation"""
    generated_images = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate how many images to actually save based on ratio
    total_images = len(captions)
    images_to_save = max(1, int(total_images * save_images_ratio))
    save_indices = np.linspace(0, total_images-1, images_to_save, dtype=int)
    
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning([""])
            
            for i, caption in enumerate(tqdm(captions, desc="Generating images")):
                c = model.get_learned_conditioning([caption])
                shape = [4, opt.H//8, opt.W//8]
                
                samples_ddim, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta
                )
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                
                # Save generated image only if it's in the save_indices
                if i in save_indices:
                    x_sample = x_samples_ddim[0]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    output_path = os.path.join(output_dir, f"generated_{i:05d}.png")
                    Image.fromarray(x_sample.astype(np.uint8)).save(output_path)
                    generated_images.append(output_path)
                else:
                    # Still need to create a placeholder path for evaluation metrics
                    # We'll use the tensor directly for metrics calculation
                    x_sample = x_samples_ddim[0]
                    generated_images.append(x_sample)
    
    return generated_images


def evaluate_text_to_image(args):
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    print("Loading Flickr30k dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = Flickr30kDataset(
        root_dir=os.path.join(args.dataset_root, "images"), 
        captions_file=os.path.join(args.dataset_root, "captions.txt"),
        transform=transform
    )
    
    # Sample subset for evaluation if dataset is large
    if args.max_samples and len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset.data = [dataset.data[i] for i in indices]
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load text-to-image model
    print("Loading text-to-image model...")
    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.model_path)
    
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    # Generate images
    print("Generating images from captions...")
    captions = [item['caption'] for item in dataset.data]
    
    # Create options object for generation
    class GenerationOpt:
        def __init__(self):
            self.ddim_steps = args.ddim_steps
            self.H = args.H
            self.W = args.W
            self.scale = args.scale
            self.ddim_eta = args.ddim_eta
    
    opt = GenerationOpt()
    generated_images = generate_images_for_evaluation(
        model, sampler, captions, opt, args.output_dir, args.save_images_ratio
    )
    
    # Collect real images
    real_images = [item['image_path'] for item in dataset.data]
    
    # Initialize evaluators
    print("Initializing evaluators...")
    fid_calc = FIDCalculator(device)
    pr_calc = PrecisionRecallCalculator(k=3)
    clip_calc = CLIPScoreCalculator(device)
    
    # Compute FID Score
    print("Computing FID Score...")
    real_features = fid_calc.get_features(real_images)
    generated_features = fid_calc.get_features(generated_images)
    fid_score = fid_calc.calculate_fid(real_features, generated_features)
    
    # Compute Precision and Recall
    print("Computing Precision and Recall...")
    precision, recall = pr_calc.compute_precision_recall(real_features, generated_features)
    
    # Compute CLIP Score
    print("Computing CLIP Score...")
    clip_scores = clip_calc.compute_clip_score(generated_images, captions)
    mean_clip_score = np.mean(clip_scores)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset_root}")
    print(f"Number of samples: {len(captions)}")
    print(f"Images saved ratio: {args.save_images_ratio}")
    print(f"Generated images saved to: {args.output_dir}")
    print("\n" + "-"*30)
    print("METRICS:")
    print("-"*30)
    print(f"FID Score: {fid_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"CLIP Score (mean): {mean_clip_score:.4f}")
    print(f"CLIP Score (std): {np.std(clip_scores):.4f}")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("FLICKR30K TEXT-TO-IMAGE EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {args.dataset_root}\n")
        f.write(f"Number of samples: {len(captions)}\n")
        f.write(f"Images saved ratio: {args.save_images_ratio}\n")
        f.write(f"Model config: {args.config_path}\n")
        f.write(f"Model checkpoint: {args.model_path}\n")
        f.write("\nMETRICS:\n")
        f.write("-"*30 + "\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"CLIP Score (mean): {mean_clip_score:.4f}\n")
        f.write(f"CLIP Score (std): {np.std(clip_scores):.4f}\n")
        f.write(f"\nCAPTIONS AND CORRESPONDING IMAGES:\n")
        f.write("-"*50 + "\n")
        
        # Write caption and image correspondence
        for i, (caption, clip_score) in enumerate(zip(captions, clip_scores)):
            f.write(f"\nSample {i+1:03d}:\n")
            f.write(f"Caption: {caption}\n")
            f.write(f"Original Image: {dataset.data[i]['image_name']}\n")
            if isinstance(generated_images[i], str):
                generated_filename = os.path.basename(generated_images[i])
                f.write(f"Generated Image: {generated_filename}\n")
            else:
                f.write(f"Generated Image: generated_{i:05d}.png (not saved - controlled by save_images_ratio)\n")
            f.write(f"CLIP Score: {clip_score:.4f}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text-to-image model on Flickr30k")
    
    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, default='./data/text2img_dataset',
                       help="Path to Flickr30k dataset root directory")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum number of samples to evaluate (for testing)")
    
    # Model arguments
    parser.add_argument("--config_path", type=str, 
                       default="configs/latent-diffusion/txt2img-1p4B-eval.yaml",
                       help="Path to model config file")
    parser.add_argument("--model_path", type=str,
                       default="models/ldm/text2img-large/model.ckpt",
                       help="Path to model checkpoint")
    
    # Generation arguments
    parser.add_argument("--ddim_steps", type=int, default=100,
                       help="Number of DDIM sampling steps")
    parser.add_argument("--plms", action='store_true',
                       help="Use PLMS sampling")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                       help="DDIM eta parameter")
    parser.add_argument("--H", type=int, default=256,
                       help="Image height")
    parser.add_argument("--W", type=int, default=256,
                       help="Image width")
    parser.add_argument("--scale", type=float, default=7.5,
                       help="Unconditional guidance scale")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="evaluation/text2img",
                       help="Directory to save generated images and results")
    parser.add_argument("--save_images_ratio", type=float, default=1.0,
                       help="Ratio of images to actually save to disk (0.0-1.0). For testing, use smaller values like 0.1")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_text_to_image(args)