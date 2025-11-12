# How `get_dataloaders` in utils.py Handles Memory

## Key Difference: WebDataset vs HDF5

The `get_dataloaders` function in `utils.py` uses **WebDataset** (tar files) instead of HDF5 files. This is a fundamentally different approach to memory management.

---

## Memory Management Strategy

### 1. **Streaming Data Format (WebDataset)**

```python
train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
    .shuffle(500, initial=500, rng=random.Random(42))\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", ...)\
    .to_tuple(*to_tuple)\
    .batched(batch_size, partial=True)\
    .with_epoch(num_worker_batches)
```

**How it works:**
- **WebDataset** reads from `.tar` files (not HDF5)
- Data is **streamed** from disk, not loaded entirely into memory
- Each tar file contains multiple samples as separate files
- Data is decoded on-the-fly as it's read

### 2. **Lazy Decoding**

```python
.decode("torch")
```
- Images and voxels are decoded **only when needed**
- Not pre-loaded into memory
- Decoding happens in the data pipeline, not upfront

### 3. **Batching in Pipeline**

```python
.batched(batch_size, partial=True)
```
- Batching happens **before** DataLoader
- Only one batch is in memory at a time
- `partial=True` allows incomplete final batches

### 4. **Limited Worker Processes**

```python
train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, num_workers=1, shuffle=False)
```

**Key observation: `num_workers=1`**
- Only **one worker process** is used
- This limits parallel data loading but also limits memory usage
- Each worker would cache some data, so fewer workers = less memory

### 5. **Caching Strategy**

```python
cache_dir="/tmp/wds-cache"
```
- Optional caching directory for downloaded data
- If data is already downloaded, it's read from disk
- Cache is on disk, not in RAM

---

## Memory Usage Comparison

### HDF5 Approach (dataloader.py):
- **File handles**: ~200 MB (40 session files open)
- **Active data**: ~9 MB per trial, ~300 MB per batch
- **Total**: ~1-2 GB during training
- **Issue**: All 40 HDF5 files could theoretically be open simultaneously

### WebDataset Approach (utils.py):
- **File handles**: Minimal (tar files opened sequentially)
- **Active data**: Only current batch in memory
- **Streaming buffer**: ~500 samples shuffled (configurable)
- **Total**: **~500 MB - 1 GB** (typically less than HDF5)

---

## Why WebDataset is More Memory Efficient

### 1. **Sequential File Access**
- Tar files are read sequentially, not randomly accessed
- Only one tar file needs to be open at a time
- No need to keep multiple file handles open

### 2. **No Random Access Overhead**
- HDF5 allows random access to any trial → requires file handles to stay open
- WebDataset streams data → can close files after reading

### 3. **Pipeline-Based Processing**
```
Tar File → Shuffle Buffer (500 samples) → Decode → Batch → DataLoader
```
- Each stage processes data incrementally
- Only the shuffle buffer (500 samples) is held in memory
- Everything else is streamed

### 4. **Smaller Memory Footprint Per Sample**
- WebDataset stores data as individual files in tar archives
- Can be more efficiently compressed
- Only decompresses what's needed

---

## Memory Breakdown for WebDataset

### During Training:

1. **Shuffle Buffer** (Line 313):
   ```python
   .shuffle(500, initial=500, ...)
   ```
   - Holds 500 samples in memory for shuffling
   - Per sample: ~9 MB (fMRI) + ~0.5 MB (image) = ~9.5 MB
   - **Buffer size: 500 × 9.5 MB = ~4.75 GB** (worst case)
   - But this is **compressed/encoded** data, so actual memory is less

2. **Active Batch**:
   - One batch being processed: `batch_size × 9.5 MB`
   - Example: batch_size=32 → **~300 MB**

3. **Decoded Data**:
   - Only the current batch is decoded
   - Previous batches are garbage collected

4. **File Handles**:
   - Minimal - tar files opened/closed as needed
   - **~10-50 MB** total

### Total Memory Usage:
- **Shuffle buffer**: ~2-3 GB (compressed data)
- **Active batches**: ~300 MB
- **Overhead**: ~100 MB
- **Total: ~2.5-3.5 GB** (but can be optimized)

---

## Key Memory Optimizations in WebDataset

### 1. **Resampling Instead of Full Dataset**
```python
resampled=True
```
- Doesn't load entire dataset into memory
- Samples are generated on-the-fly
- Can handle infinite datasets

### 2. **Epoch-Based Batching**
```python
.with_epoch(num_worker_batches)
```
- Limits how many batches are processed per epoch
- Prevents memory buildup over long training runs

### 3. **Single Worker**
```python
num_workers=1
```
- Avoids memory duplication across multiple workers
- Each worker would cache data independently
- Single worker = single cache

---

## Comparison Table

| Aspect | HDF5 (dataloader.py) | WebDataset (utils.py) |
|--------|---------------------|----------------------|
| **File Format** | HDF5 (binary) | Tar files (archive) |
| **Access Pattern** | Random access | Sequential streaming |
| **File Handles** | Multiple (40 files) | Single (one tar at a time) |
| **Memory per Batch** | ~300 MB | ~300 MB |
| **Overhead** | ~200 MB (file handles) | ~2-3 GB (shuffle buffer) |
| **Total Memory** | ~1-2 GB | ~2.5-3.5 GB |
| **Scalability** | Limited by open files | Better for large datasets |
| **Random Access** | Fast (direct indexing) | Slower (must stream) |

---

## When to Use Each Approach

### Use HDF5 (dataloader.py) when:
- You need random access to specific trials
- Dataset fits in available memory
- You want minimal memory overhead
- Working with pre-processed HDF5 files

### Use WebDataset (utils.py) when:
- Dataset is too large for memory
- You need to stream data from remote sources
- You want better scalability
- Data is already in WebDataset format
- You need distributed training support

---

## Current Usage in Codebase

Looking at `Train_MindEye.py` (line 253), the code is currently using:
```python
train_dl, val_dl, num_train, num_val = dataloader.get_dataloaders(...)
```

This uses the **HDF5-based approach** from `dataloader.py`, not the WebDataset approach from `utils.py`.

The WebDataset version in `utils.py` is commented out (lines 236-252), suggesting it was used previously but switched to HDF5 for the current implementation.

---

## Summary

The `get_dataloaders` in `utils.py` handles memory by:
1. **Streaming** data from tar files instead of random access
2. **Decoding on-the-fly** rather than pre-loading
3. **Using a shuffle buffer** (500 samples) instead of loading entire dataset
4. **Single worker** to avoid memory duplication
5. **Pipeline-based processing** where only active batches are in memory

**Trade-off**: Slightly higher memory usage (2.5-3.5 GB vs 1-2 GB) but better scalability and ability to handle datasets that don't fit in memory.

