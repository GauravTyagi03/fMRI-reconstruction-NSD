import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
save_dir = "../img"
os.makedirs(save_dir, exist_ok=True)

# Load the files
all_brain_recons = torch.load('mindeye_test1_recons_img2img1_4samples.pt')
all_images = torch.load('all_images.pt')

# Calculate data structure
num_samples = all_images.shape[0]  # Should be 2 (train + val)
recons_per_sample = all_brain_recons.shape[0] // num_samples
train_sample_idx = 0
val_sample_idx = 0

print(f"Loaded {num_samples} ground truth images")
print(f"Loaded {all_brain_recons.shape[0]} reconstructions ({recons_per_sample} per sample)")

# Resize for display (optional)
imsize = 256
all_images = transforms.Resize((imsize, imsize))(all_images)
all_brain_recons = transforms.Resize((imsize, imsize))(all_brain_recons)

# Create visualization for each split
splits = [
    ('train', 0, train_sample_idx),
    ('val', 1, val_sample_idx)
]

for split_name, img_idx, original_idx in splits:
    # Calculate reconstruction indices for this sample
    recon_start_idx = img_idx * recons_per_sample
    recon_end_idx = (img_idx + 1) * recons_per_sample

    # Create figure with 1 + recons_per_sample columns
    fig, axes = plt.subplots(1, 1 + recons_per_sample, figsize=(5 * (1 + recons_per_sample), 5))

    # Plot ground truth
    axes[0].imshow(transforms.ToPILImage()(all_images[img_idx]))
    axes[0].set_title(f"{split_name.capitalize()} Ground Truth (idx: {original_idx})", fontsize=12)
    axes[0].axis('off')

    # Plot all reconstructions
    for i in range(recons_per_sample):
        recon_idx = recon_start_idx + i
        axes[i + 1].imshow(transforms.ToPILImage()(all_brain_recons[recon_idx]))
        axes[i + 1].set_title(f"Recon {i + 1}", fontsize=12)
        axes[i + 1].axis('off')

    # Save figure
    save_path = os.path.join(save_dir, f"{split_name}_sample_{original_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"Saved {save_path}")
