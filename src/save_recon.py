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

# Resize for display (optional)
imsize = 256
all_images = transforms.Resize((imsize, imsize))(all_images)
all_brain_recons = transforms.Resize((imsize, imsize))(all_brain_recons)

# Display and save specific samples
# Updated to use valid indices (0-49) since only 50 samples were processed
sample_indices = [0, 10, 20, 30, 44]  # Example indices

for idx in sample_indices:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(transforms.ToPILImage()(all_images[idx]))
    ax1.set_title("Ground Truth")
    ax1.axis('off')

    ax2.imshow(transforms.ToPILImage()(all_brain_recons[idx]))
    ax2.set_title("Reconstruction")
    ax2.axis('off')

    # Save the figure instead of showing it
    save_path = os.path.join(save_dir, f"sample_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {save_path}")
