# Memory Analysis: CPU RAM vs GPU Memory for DataLoading

## Understanding Memory Usage

### CPU RAM (System Memory) - 64GB Requested
**Used for:**
- Loading data from HDF5 files (disk → RAM)
- DataLoader worker processes
- PyTorch tensors on CPU
- Image preprocessing (PIL, transforms)
- NumPy arrays
- File handles and metadata

### GPU Memory (VRAM) - 80GB Available (H100)
**Used for:**
- Model parameters
- Model activations during forward/backward pass
- Batches moved to GPU with `.to(device)`
- Gradients
- Optimizer states
- CUDA kernels and temporary buffers

---

## Memory Flow in Your Pipeline

```
Disk (HDF5 files)
    ↓ [Read by DataLoader workers]
CPU RAM (System Memory) ← YOU HAVE 64GB HERE
    ↓ [.to(device) in training loop]
GPU Memory (VRAM) ← YOU HAVE 80GB HERE
    ↓ [Model processing]
GPU Memory (activations, gradients)
```

**Key Point:** DataLoader loads data into **CPU RAM first**, then you explicitly move it to GPU with `.to(device)`.

---

## Memory Calculation for Your Setup

### Assumptions:
- Batch size: 32 (from Train_MindEye.py)
- num_workers: 4 (from dataloader.py line 101)
- fMRI shape: (150, 150, 100) = 2,250,000 voxels per trial
- Image shape: (425, 425, 3) = 541,875 pixels per image
- DataLoader prefetch_factor: 2 (PyTorch default)

### Per Trial Memory (CPU RAM):
1. **fMRI data (float32)**: 2,250,000 × 4 bytes = **9 MB**
2. **Image (uint8 → float32)**: 541,875 × 3 × 4 bytes = **6.5 MB**
3. **Transformed tensors**: ~16 MB (with transforms)
4. **Total per trial**: **~16 MB**

### Per Batch Memory (CPU RAM):
- **32 trials × 16 MB = 512 MB per batch**

### DataLoader Worker Memory:
With `num_workers=4` and `prefetch_factor=2` (default):
- Each worker prefetches 2 batches
- **4 workers × 2 batches × 512 MB = 4 GB** (just for data)
- Plus worker process overhead: **~500 MB per worker**
- **Total worker memory: ~6 GB**

### File Handle Memory:
- Open HDF5 files: ~5-10 MB per file
- With lazy loading, potentially 10-20 files open: **~100-200 MB**

### Total CPU RAM Usage:
- **DataLoader workers**: ~6 GB
- **File handles**: ~0.2 GB
- **Main process**: ~2 GB (Python, PyTorch, models on CPU)
- **Buffer/overhead**: ~2 GB
- **Total: ~10-12 GB** (well within 64 GB!)

---

## GPU Memory Usage

### Per Batch on GPU:
- **fMRI batch**: 32 × 2,250,000 × 4 bytes = **288 MB**
- **Image batch**: 32 × 3 × 425 × 425 × 4 bytes = **69 MB**
- **Total per batch**: **~357 MB**

### Model Memory (estimated):
- Diffusion Prior model: **~5-10 GB**
- CLIP extractor: **~1-2 GB**
- Optimizer states: **~10-20 GB** (Adam stores 2x model params)
- Activations during training: **~5-10 GB**
- **Total model: ~20-40 GB**

### Total GPU Memory:
- **Model + optimizer**: ~30 GB
- **Active batches**: ~0.5 GB
- **Activations/gradients**: ~10 GB
- **CUDA overhead**: ~5 GB
- **Total: ~45-50 GB** (well within 80 GB!)

---

## Will Lazy Loading Work with 64GB RAM?

**YES, absolutely!** Here's why:

1. **DataLoader only loads batches**: Not entire dataset
2. **Workers prefetch limited batches**: 2 batches per worker max
3. **Lazy file opening**: Only opens files when needed
4. **Garbage collection**: Old batches are freed after use

**Estimated usage: ~10-12 GB out of 64 GB = 15-20% utilization**

**You have plenty of headroom!**

---

## How to Monitor Memory Usage

### 1. Monitor CPU RAM During Training

Add this to your training script:

```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"CPU RAM: {mem_info.rss / 1024**3:.2f} GB")
    
    # System-wide
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB ({mem.percent}%)")

# Call in training loop
for epoch in range(num_epochs):
    print_memory_usage()
    for batch in train_dl:
        # ... training code
```

### 2. Monitor GPU Memory

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Call in training loop
for epoch in range(num_epochs):
    print_gpu_memory()
    for batch in train_dl:
        # ... training code
```

### 3. Use SLURM Monitoring

In your SLURM script, add:

```bash
# Monitor memory in output file
srun --mem=64G python -u Train_MindEye.py ... 2>&1 | tee training.log
```

Then check the output file for memory usage.

### 4. Real-time Monitoring Script

Create `monitor_memory.sh`:

```bash
#!/bin/bash
# Run this in another terminal while training

watch -n 1 '
echo "=== CPU Memory ==="
free -h
echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""
echo "=== Process Memory ==="
ps aux | grep Train_MindEye | grep -v grep | awk "{print \"PID: \" \$2 \", RSS: \" \$6/1024 \" MB\"}"
'
```

---

## Techniques to Prevent OOM

### 1. Limit DataLoader Workers

If you run into CPU RAM issues, reduce workers:

```python
# In get_dataloaders
num_workers = min(4, os.cpu_count() // 2)  # Use fewer workers
```

### 2. Reduce Prefetch Factor

```python
train_dl = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    prefetch_factor=1,  # Reduce from default 2
    persistent_workers=False  # Don't keep workers alive between epochs
)
```

### 3. Close Files After Use (Fix Current Implementation)

Modify `__getitem__` to close files immediately:

```python
def __getitem__(self, idx):
    session, trial_in_session, global_trial = self.trial_map[idx]
    
    # Open, read, close immediately
    fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
    with h5py.File(fmri_path, "r") as f:
        fmri_data = f["betas"][trial_in_session].astype(np.float32)
    
    # ... rest of code
```

### 4. Use Pin Memory (for GPU transfer speed)

```python
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Faster CPU→GPU transfer (uses ~batch_size extra RAM)
    pin_memory_device='cuda'  # Pin directly to GPU (if supported)
)
```

**Trade-off**: `pin_memory=True` uses extra CPU RAM (~batch_size × 2) but speeds up GPU transfer.

### 5. Reduce Batch Size (if needed)

```python
# If you hit memory limits, reduce batch size
batch_size = 16  # Instead of 32
```

### 6. Use Gradient Accumulation

Instead of reducing batch size, accumulate gradients:

```python
accumulation_steps = 2  # Effective batch size = 32 * 2 = 64

for i, batch in enumerate(train_dl):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 7. Clear Cache Periodically

```python
import gc

for epoch in range(num_epochs):
    for batch in train_dl:
        # ... training code
        del batch  # Explicit deletion
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()  # Clear GPU cache
```

### 8. Monitor and Alert

Add memory checks that raise warnings:

```python
import psutil

def check_memory(threshold_gb=50):
    mem = psutil.virtual_memory()
    if mem.used / 1024**3 > threshold_gb:
        print(f"WARNING: High memory usage: {mem.used / 1024**3:.2f} GB")
        return True
    return False

# In training loop
if check_memory(50):
    # Reduce workers, clear cache, etc.
    pass
```

---

## Recommended DataLoader Configuration

Based on your 64GB RAM, here's an optimized configuration:

```python
def get_dataloaders(batch_size,
                    fmri_dir,
                    stimulus_path,
                    train_idx_path,
                    val_idx_path,
                    transform_image=None,
                    transform_fmri=None,
                    num_workers=4,
                    pin_memory=True,  # Faster GPU transfer
                    prefetch_factor=2,  # Default is fine
                    persistent_workers=True):  # Keep workers alive

    train_ds = NSDDataset(fmri_dir, stimulus_path, train_idx_path, 
                         transform_image=transform_image, 
                         transform_fmri=transform_fmri)
    val_ds = NSDDataset(fmri_dir, stimulus_path, val_idx_path,
                       transform_image=transform_image, 
                       transform_fmri=transform_fmri)

    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_dl, val_dl, len(train_ds), len(val_ds)
```

---

## Memory Monitoring Checklist

Before training, verify:

- [ ] Check available RAM: `free -h`
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Add memory monitoring to training script
- [ ] Set up real-time monitoring (separate terminal)
- [ ] Test with small batch first
- [ ] Monitor first few epochs closely
- [ ] Check SLURM output for OOM errors

---

## Expected Memory Usage Summary

| Component | CPU RAM | GPU Memory |
|-----------|---------|------------|
| **DataLoader** | ~6 GB | 0 GB |
| **File handles** | ~0.2 GB | 0 GB |
| **Main process** | ~2 GB | 0 GB |
| **Model (CPU)** | ~2 GB | 0 GB |
| **Model (GPU)** | 0 GB | ~30 GB |
| **Active batches** | ~0.5 GB | ~0.5 GB |
| **Gradients/activations** | 0 GB | ~10 GB |
| **Total** | **~10-12 GB** | **~40-50 GB** |
| **Available** | **64 GB** | **80 GB** |
| **Utilization** | **15-20%** | **50-60%** |

**Conclusion: You have plenty of headroom! Lazy loading will work fine.**

---

## Quick Test Script

Create `test_memory.py` to verify before full training:

```python
import torch
from dataloader import get_dataloaders
import psutil
import os

# Test with small dataset
train_dl, val_dl, num_train, num_val = get_dataloaders(
    batch_size=32,
    fmri_dir="...",
    stimulus_path="...",
    train_idx_path="...",
    val_idx_path="...",
    num_workers=4
)

# Check initial memory
process = psutil.Process(os.getpid())
print(f"Initial RAM: {process.memory_info().rss / 1024**3:.2f} GB")

# Load a few batches
for i, batch in enumerate(train_dl):
    print(f"After batch {i+1}: {process.memory_info().rss / 1024**3:.2f} GB")
    if i >= 10:  # Test 10 batches
        break

print(f"Final RAM: {process.memory_info().rss / 1024**3:.2f} GB")
print("Memory test complete!")
```

