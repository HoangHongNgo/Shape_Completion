from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import SSCM_dataloader
import numpy as np

gn_root = '/media/dsp520/Grasp_2T/graspnet'
camera = 'realsense'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_set = SSCM_dataloader.SSCM_dataset(
            gn_root, camera, rgb_only=False, pred_depth=True)
train_dataloader = DataLoader(
        training_set, batch_size=12, num_workers=8)

# First pass: collect depth statistics completely on GPU
depth_values_sum = torch.tensor(0.0, device=device)
depth_values_squared_sum = torch.tensor(0.0, device=device)
valid_pixel_count = torch.tensor(0, device=device)

loader = tqdm(train_dataloader, desc="Computing statistics on GPU")
for idx, data in enumerate(loader):
    torch.cuda.empty_cache()
    rgbd = data["rgbd"].to(device)
    depth = rgbd[:, 3, :, :]
    
    # Skip invalid/nan values
    valid_mask = torch.isfinite(depth) & (depth > 0)
    
    # Update running statistics on GPU
    valid_depths = depth[valid_mask]
    if valid_depths.numel() > 0:
        depth_values_sum += torch.sum(valid_depths)
        depth_values_squared_sum += torch.sum(valid_depths * valid_depths)
        valid_pixel_count += valid_depths.numel()

# Compute mean and standard deviation on GPU
if valid_pixel_count > 0:
    depth_mean = depth_values_sum / valid_pixel_count
    depth_variance = (depth_values_squared_sum / valid_pixel_count) - (depth_mean * depth_mean)
    depth_std = torch.sqrt(depth_variance)
    
    # Define outlier thresholds on GPU
    lower_bound = depth_mean - 3 * depth_std
    upper_bound = depth_mean + 3 * depth_std
    
    print(f"Depth statistics: Mean = {depth_mean.item():.2f}, Std = {depth_std.item():.2f}")
    print(f"Outlier threshold: [{lower_bound.item():.2f}, {upper_bound.item():.2f}]")
else:
    print("No valid depth values found!")
    
# Second pass: find min/max excluding outliers (entirely on GPU)
min_depth = torch.tensor(float('inf'), device=device)
max_depth = torch.tensor(float('-inf'), device=device)

# Reset dataloader
train_dataloader = DataLoader(
        training_set, batch_size=32, num_workers=8)
        
loader = tqdm(train_dataloader, desc="Finding min/max on GPU (excluding outliers)")
for idx, data in enumerate(loader):
    torch.cuda.empty_cache()
    rgbd = data["rgbd"].to(device)
    depth = rgbd[:, 3, :, :]
    
    # Skip invalid/nan values and outliers
    valid_mask = torch.isfinite(depth) & (depth > 0) & (depth >= lower_bound) & (depth <= upper_bound)
    
    if torch.sum(valid_mask) > 0:
        batch_min = torch.min(depth[valid_mask])
        batch_max = torch.max(depth[valid_mask])
        
        min_depth = torch.min(min_depth, batch_min)
        max_depth = torch.max(max_depth, batch_max)
    
    # Update the progress bar with current min/max values
    loader.set_postfix(min_depth=min_depth.item(), max_depth=max_depth.item())

print(f"Minimum depth value (excluding outliers): {min_depth.item()}")
print(f"Maximum depth value (excluding outliers): {max_depth.item()}")