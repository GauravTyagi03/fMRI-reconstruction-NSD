# Memory and Multi-Worker Issues in dataloader.py

## Critical Finding: The Current Implementation Has Issues

After analyzing the code, I found that **the current implementation does NOT prevent multiple session files from being open simultaneously**, and there are **potential multi-worker conflicts**.

---

## Issue 1: Multiple Session Files Stay Open

### Current Behavior:

```python
# Line 51: Dictionary to store open file handles
self.fmri_files = {}

# Lines 63-65: Lazy opening (but files never close!)
if session not in self.fmri_files:
    fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
    self.fmri_files[session] = h5py.File(fmri_path, "r")
```

### Problem:
- Files are opened **lazily** (good!)
- But once opened, they **stay open** until the dataset is destroyed
- If training accesses sessions 1, 5, 10, 15, 20, 25, 30, 35, 40, **all 9 files remain open**
- With shuffled data accessing random sessions, you could easily have **10-20 files open simultaneously**

### Memory Impact:
- Each open HDF5 file handle: **~5-10 MB** (metadata, not full data)
- 20 open files: **~100-200 MB** just for file handles
- This is manageable but not optimal

### Why This Happens:
- Shuffled DataLoader accesses trials in random order
- Random order → random sessions
- Each new session opens a new file
- Files accumulate in `self.fmri_files` dictionary
- No mechanism to close unused files

---

## Issue 2: Multi-Worker File Handle Multiplication

### Current Behavior:

```python
# Line 106: DataLoader with multiple workers
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
```

### How PyTorch DataLoader Works with Workers:
1. **Each worker is a separate process**
2. **Each worker gets its own copy of the dataset**
3. **Each worker has its own `self.fmri_files` dictionary**
4. **Workers operate independently**

### Problem Scenario:

If `num_workers=4`:
- **Worker 0** accesses sessions [1, 5, 10, 15] → opens 4 files
- **Worker 1** accesses sessions [2, 7, 12, 18] → opens 4 files  
- **Worker 2** accesses sessions [3, 8, 13, 20] → opens 4 files
- **Worker 3** accesses sessions [4, 9, 14, 25] → opens 4 files

**Total: 16 file handles open simultaneously** (across all workers)

### Worse Case:
If all workers access different sessions:
- **4 workers × 10 sessions each = 40 file handles**
- All 40 session files could be open at once!

### Memory Impact:
- **40 files × 5-10 MB = 200-400 MB** just for file handles
- Still manageable, but inefficient

---

## Issue 3: No File Handle Clashing (But No Coordination Either)

### Good News:
- HDF5 files opened in **read-only mode** (`"r"`)
- Multiple processes can read the same file simultaneously
- **No data corruption risk**
- **No locking needed** for read-only access

### Bad News:
- **No coordination** between workers
- Each worker independently opens files
- **Redundant file handles** (multiple workers opening same file)
- **No shared caching** across workers

### Example:
If Worker 0 and Worker 1 both need session 5:
- Worker 0 opens `betas_session05.hdf5` → File handle A
- Worker 1 opens `betas_session05.hdf5` → File handle B
- **Same file opened twice** (wasteful but safe)

---

## Issue 4: Code Bugs in `__getitem__`

Looking at lines 57-66, there are **undefined variables**:

```python
def __getitem__(self, idx):
    fmri_idx, img_idx = self.trial_idx.T  # ❌ trial_idx is 1D, not 2D!
    session_idx = fmri_idx % 750
    session = fmri_idx // 750 + 1
    
    # ...
    
    fmri_data = self.fmri_files[session]["betas"][trial_in_session]  # ❌ trial_in_session not defined!
    # ...
    stim_id = int(self.trial_idx[global_trial])  # ❌ global_trial not defined!
```

**This code will crash!** It looks like it was partially refactored but not completed.

---

## Solutions

### Solution 1: Close Unused Files (LRU Cache)

Implement a Least Recently Used (LRU) cache to close files that haven't been accessed recently:

```python
from collections import OrderedDict
import time

class NSDDataset(Dataset):
    def __init__(self, ..., max_open_files=5):
        # ...
        self.max_open_files = max_open_files
        self.fmri_files = OrderedDict()  # Maintains insertion order
        self.file_access_times = {}  # Track last access time
    
    def __getitem__(self, idx):
        session, trial_in_session, global_trial = self.trial_map[idx]
        
        # Close least recently used file if at limit
        if len(self.fmri_files) >= self.max_open_files:
            if session not in self.fmri_files:
                # Remove oldest file
                oldest_session = next(iter(self.fmri_files))
                self.fmri_files[oldest_session].close()
                del self.fmri_files[oldest_session]
                del self.file_access_times[oldest_session]
        
        # Open file if needed
        if session not in self.fmri_files:
            fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
            self.fmri_files[session] = h5py.File(fmri_path, "r")
        
        # Update access time and move to end (most recently used)
        self.file_access_times[session] = time.time()
        self.fmri_files.move_to_end(session)
        
        # ... rest of code
```

**Benefits:**
- Limits open files to `max_open_files` (e.g., 5)
- Automatically closes unused files
- Keeps frequently accessed files open

**Drawbacks:**
- Slight overhead from opening/closing files
- Need to track access times

### Solution 2: Use File Locking/Sharing (Advanced)

Use HDF5's SWMR (Single Writer Multiple Reader) mode or file locking to share file handles across workers. However, this is complex and may not be necessary for read-only access.

### Solution 3: Open Files Per Request (Simplest)

Open and close files on every access (no caching):

```python
def __getitem__(self, idx):
    session, trial_in_session, global_trial = self.trial_map[idx]
    
    # Open, read, close immediately
    fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
    with h5py.File(fmri_path, "r") as f:
        fmri_data = f["betas"][trial_in_session].astype(np.float32)
    
    # ... rest of code
```

**Benefits:**
- No file handle accumulation
- Simple and safe
- Works perfectly with multiple workers

**Drawbacks:**
- **Slower** - file open/close overhead on every access
- HDF5 file opening is relatively fast, but still adds latency

### Solution 4: Fix the Code Bugs First!

Before addressing memory issues, **fix the bugs in `__getitem__`**:

```python
def __getitem__(self, idx):
    session, trial_in_session, global_trial = self.trial_map[idx]  # ✅ Use trial_map
    
    # open fMRI file lazily
    if session not in self.fmri_files:
        fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
        self.fmri_files[session] = h5py.File(fmri_path, "r")
    fmri_data = self.fmri_files[session]["betas"][trial_in_session].astype(np.float32)  # ✅ Now defined
    
    # ... rest of code using global_trial ✅
```

---

## Recommended Approach

**For your use case, I recommend Solution 3 (open/close per request)** because:

1. **Simplicity**: No complex caching logic
2. **Safety**: No file handle accumulation
3. **Multi-worker friendly**: Each worker opens/closes independently
4. **Performance**: HDF5 file opening is fast (~1-5ms)
5. **Memory efficient**: Zero file handles when not actively reading

The performance hit from opening files is minimal compared to:
- Reading 150×150×100 float32 array (~9 MB)
- Image decoding and transforms
- Network forward/backward passes

---

## Current State Summary

| Issue | Current Behavior | Impact |
|-------|------------------|--------|
| **Multiple files open** | ✅ Files stay open once accessed | Medium (100-400 MB) |
| **Multi-worker handles** | ✅ Each worker opens independently | Medium (multiplies handles) |
| **File clashing** | ✅ No clashing (read-only safe) | None (safe) |
| **Code bugs** | ❌ Undefined variables | **CRITICAL (will crash)** |

---

## Action Items

1. **URGENT**: Fix the bugs in `__getitem__` (lines 58-60, 66, 73)
2. **RECOMMENDED**: Implement file closing strategy (Solution 3 is simplest)
3. **OPTIONAL**: Add LRU cache if you want to optimize for performance

