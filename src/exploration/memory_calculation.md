# Memory Calculation for NSD Session Files

## Given Dimensions
Each session file: `(750, 150, 150, 100)`
- 750 trials per session
- 150 × 150 × 100 = 2,250,000 voxels per trial
- Total elements per session: **750 × 150 × 150 × 100 = 1,687,500,000 elements**

## Memory Per Session

### If loaded into RAM (float32):
- Data type: `float32` (4 bytes per element, as seen in line 66: `.astype(np.float32)`)
- Memory per session: **1,687,500,000 × 4 bytes = 6,750,000,000 bytes**
- **= 6.75 GB** (decimal) or **~6.29 GiB** (binary)

### On Disk (HDF5 file):
- HDF5 files are often compressed, so disk size is typically **smaller**
- Compression ratio depends on data characteristics (usually 2-5x compression for fMRI data)
- Estimated disk size: **~1.5-3 GB per session** (compressed)

## Memory for 40 Sessions

### If all loaded into RAM simultaneously:
- **40 × 6.75 GB = 270 GB** (decimal)
- **~251.6 GiB** (binary)

### On Disk:
- **40 × ~2 GB = ~80 GB** (compressed, estimated)

## Important: Lazy Loading in NSDDataset

**The dataset does NOT load all sessions into RAM!**

Looking at the code:
- **Line 51**: `self.fmri_files = {}` - Empty dictionary initially
- **Lines 63-65**: Files are opened **lazily** (only when first accessed)
- **Line 66**: Only **one trial** is loaded at a time: `f["betas"][trial_in_session]`

### Actual Memory Usage During Training:

1. **File Handles**: 
   - Each open HDF5 file handle: **~few MB** (just metadata)
   - If all 40 sessions accessed: **~100-200 MB** for handles

2. **Active Data in Memory**:
   - Per trial: **150 × 150 × 100 × 4 bytes = 9,000,000 bytes = 9 MB**
   - Per batch (e.g., batch_size=32): **32 × 9 MB = 288 MB**
   - Plus images: **32 × 425 × 425 × 3 × 1 byte = ~17 MB**
   - **Total per batch: ~305 MB**

3. **DataLoader with num_workers**:
   - Each worker process may cache some data
   - With `num_workers=4`: **~1-2 GB total** (depending on prefetching)

## Summary

| Scenario | Memory Usage |
|----------|-------------|
| **Single session in RAM** | 6.75 GB |
| **40 sessions in RAM** | 270 GB |
| **40 sessions on disk** | ~80 GB (compressed) |
| **Actual training (lazy loading)** | ~1-2 GB (only active batches) |

## Why Lazy Loading is Critical

Without lazy loading, you'd need **270 GB of RAM** to load all sessions. With the current implementation:
- Only file handles are kept open (~200 MB)
- Only requested trials are loaded (~9 MB per trial)
- Total memory: **~1-2 GB** during training

This is why the `NSDDataset` class uses lazy loading - it makes the dataset feasible to work with on standard hardware!

