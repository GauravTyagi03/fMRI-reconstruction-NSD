# NSDDataset Class - Complete Explanation

## Overview
The `NSDDataset` class is a PyTorch Dataset that loads fMRI brain data and corresponding stimulus images from the Natural Scenes Dataset (NSD). It handles multiple sessions of data and maps trials to images through a complex indexing system.

---

## Component-by-Component Breakdown

### 1. **Initialization (`__init__`)**

#### Parameters:
- `fmri_dir`: Directory containing `betas_sessionXX.hdf5` files (one per session)
- `stimulus_path`: Path to `nsd_stimuli.hdf5` containing all 73k images
- `trial_idx_path`: Path to `sub01_trial_idx.npy` - **CRITICAL**: This maps 30k trials → 73k image IDs
- `sessions`: Which sessions to load (default: 1-40, can be a single int or list)
- `transform_image`: Image preprocessing (default: ToTensor)
- `transform_fmri`: fMRI preprocessing (optional)

#### Key Initialization Steps:

**Step 1: Load trial_idx mapping (Line 26)**
```python
self.trial_idx = np.load(trial_idx_path)
```
- **What it does**: Loads a NumPy array that maps each of the 30,000 trials to one of the 73,000 unique images
- **Shape**: (30000,) - one entry per trial
- **Values**: Image IDs in the range [0, 72999]
- **Why needed**: Multiple trials can show the same image (repetitions), so we need this mapping

**Step 2: Normalize sessions (Lines 31-36)**
- Converts single int → list, None → all sessions (1-40)

**Step 3: Build trial_map - INDEX BOOKKEEPING (Lines 38-46)**
```python
self.trial_map = []  # (session_num, trial_index)
for s in self.sessions:
    fmri_file = os.path.join(fmri_dir, f"betas_session{s:02d}.hdf5")
    with h5py.File(fmri_file, "r") as f:
        num_trials = f["betas"].shape[0]  # Usually 750 trials per session
    for t in range(num_trials):
        global_trial = (s - 1) * 750 + t  # Convert to global trial index
        self.trial_map.append((s, t, global_trial))
```

**Understanding trial_map:**
- **Purpose**: Maps dataset index → (session, trial_in_session, global_trial)
- **Structure**: List of tuples `(session_num, trial_in_session, global_trial)`
  - `session_num`: Which session file (1-40)
  - `trial_in_session`: Index within that session (0-749 typically)
  - `global_trial`: Global trial number across all sessions (0-29999)
- **Example**: 
  - Index 0 → Session 1, Trial 0, Global Trial 0
  - Index 750 → Session 2, Trial 0, Global Trial 750
  - Index 1000 → Session 2, Trial 250, Global Trial 1000

**Global Trial Calculation:**
```python
global_trial = (s - 1) * 750 + t
```
- Assumes 750 trials per session
- Session 1: trials 0-749 → global 0-749
- Session 2: trials 0-749 → global 750-1499
- Session 3: trials 0-749 → global 1500-2249
- etc.

**Step 4: Initialize lazy file handles (Lines 50-52)**
- `self.fmri_files = {}`: Dictionary to cache open HDF5 files per session
- `self.stim_file = None`: Will hold the stimulus HDF5 file handle

---

### 2. **Length (`__len__`)**

```python
def __len__(self):
    return self.num_trials
```
- Returns total number of trials across all loaded sessions
- Used by DataLoader to know dataset size

---

### 3. **Item Retrieval (`__getitem__`) - THE CORE LOGIC**

This is where everything comes together. Let's trace through step-by-step:

#### Step 1: Map dataset index to session/trial (Line 58)
```python
session, trial_in_session, global_trial = self.trial_map[idx]
```
- `idx`: The index requested by DataLoader (0 to len(dataset)-1)
- Unpacks the mapping to get:
  - Which session file to open
  - Which trial within that session
  - The global trial number (needed for image lookup)

#### Step 2: Load fMRI data (Lines 60-65)

**Lazy file opening (Lines 61-63):**
```python
if session not in self.fmri_files:
    fmri_path = os.path.join(self.fmri_dir, f"betas_session{session:02d}.hdf5")
    self.fmri_files[session] = h5py.File(fmri_path, "r")
```
- **Lazy loading**: Only opens HDF5 files when first needed
- **Caching**: Stores open file handles in `self.fmri_files` dict
- **Why**: Avoids opening all 40 session files at once (memory efficient)

**Read fMRI data (Line 64):**
```python
fmri_data = self.fmri_files[session]["betas"][trial_in_session].astype(np.float32)
```
- Accesses the "betas" dataset in the HDF5 file
- Indexes by `trial_in_session` (0-749 within that session)
- Converts to float32 for neural network compatibility

**Transform fMRI (Line 65):**
```python
fmri = self.transform_fmri(fmri_data) if self.transform_fmri else torch.tensor(fmri_data)
```
- Applies optional transform, or just converts to PyTorch tensor

#### Step 3: Load stimulus image (Lines 67-74)

**Lazy file opening (Lines 68-69):**
```python
if self.stim_file is None:
    self.stim_file = h5py.File(self.stimulus_path, "r")
```
- Opens the stimulus HDF5 file only once (first time needed)
- Caches in `self.stim_file`

**Get image ID using trial_idx_path (Line 71):**
```python
stim_id = int(self.trial_idx[global_trial])
```
- **THIS IS WHERE trial_idx_path IS USED!**
- `self.trial_idx[global_trial]` looks up which image ID corresponds to this global trial
- **Example**: If global_trial=100, and `self.trial_idx[100] = 5432`, then trial 100 shows image 5432
- This handles the fact that:
  - There are 30k trials but 73k unique images
  - Multiple trials can show the same image (repetitions)
  - The mapping is not sequential (trial 0 might show image 5000)

**Load image (Lines 72-74):**
```python
img = self.stim_file["imgBrick"][stim_id]  # [425, 425, 3], uint8
img = Image.fromarray(img)
img = self.transform_image(img)
```
- Accesses "imgBrick" dataset in stimulus HDF5
- Indexes by `stim_id` (the image ID we just looked up)
- Converts NumPy array to PIL Image
- Applies image transform (default: ToTensor)

#### Step 4: Return data (Lines 76-81)
```python
return {
    "voxels": fmri,
    "images": img,
    "trial": global_trial
}
```
- Returns dictionary with:
  - `voxels`: fMRI brain data (tensor)
  - `images`: Stimulus image (tensor)
  - `trial`: Global trial number (for tracking/debugging)

---

### 4. **Cleanup (`__del__`)**

```python
def __del__(self):
    for f in self.fmri_files.values():
        try: f.close()
        except: pass
    if self.stim_file:
        try: self.stim_file.close()
        except: pass
```
- Closes all open HDF5 file handles when dataset is destroyed
- Prevents file handle leaks
- Uses try/except because file might already be closed

---

## How `trial_idx_path` is Used Throughout

### Summary of Usage:

1. **Line 26 - Initialization**: 
   ```python
   self.trial_idx = np.load(trial_idx_path)
   ```
   - Loads the mapping array once at initialization
   - Stored as instance variable `self.trial_idx`

2. **Line 71 - Item Retrieval**:
   ```python
   stim_id = int(self.trial_idx[global_trial])
   ```
   - **Primary usage**: Maps global trial number → image ID
   - This is the critical lookup that connects fMRI trials to their images
   - Without this, you wouldn't know which image was shown during which trial

### Why trial_idx_path is Necessary:

The NSD dataset structure has:
- **30,000 trials** across 40 sessions (750 per session)
- **73,000 unique images** in the stimulus set
- **Non-sequential mapping**: Trial 0 doesn't necessarily show image 0
- **Repetitions**: Same image can appear in multiple trials

The `trial_idx.npy` file bridges this gap:
```
trial_idx[0] = 5000    → Trial 0 shows image 5000
trial_idx[1] = 1234    → Trial 1 shows image 1234
trial_idx[2] = 5000    → Trial 2 shows image 5000 (repetition!)
...
trial_idx[29999] = 67890 → Last trial shows image 67890
```

### Data Flow with trial_idx_path:

```
Dataset Index (idx)
    ↓
trial_map[idx] → (session, trial_in_session, global_trial)
    ↓
Load fMRI from: betas_session{session}.hdf5["betas"][trial_in_session]
    ↓
Lookup image: trial_idx[global_trial] → stim_id
    ↓
Load image from: nsd_stimuli.hdf5["imgBrick"][stim_id]
    ↓
Return: {voxels: fmri, images: img, trial: global_trial}
```

---

## Index Bookkeeping Summary

### Three-Level Indexing System:

1. **Dataset Index (`idx`)**: 
   - What PyTorch DataLoader uses (0 to num_trials-1)
   - Sequential across all loaded sessions

2. **Session + Trial Index**:
   - `session`: Which HDF5 file (1-40)
   - `trial_in_session`: Position within that file (0-749)

3. **Global Trial Index**:
   - `global_trial`: Sequential across all sessions (0-29999)
   - Used to lookup image ID in `trial_idx` array

### Conversion Chain:
```
idx (dataset index)
  → trial_map[idx] = (session, trial_in_session, global_trial)
  → Load fMRI: session file, trial_in_session position
  → Lookup image: trial_idx[global_trial] = stim_id
  → Load image: stim_id position in imgBrick
```

---

## Example Walkthrough

Let's trace through `dataset[1500]`:

1. **Dataset index**: 1500

2. **Lookup in trial_map**:
   ```python
   session, trial_in_session, global_trial = self.trial_map[1500]
   # Assuming session 3, trial 0:
   # session = 3
   # trial_in_session = 0
   # global_trial = (3-1)*750 + 0 = 1500
   ```

3. **Load fMRI**:
   - Open `betas_session03.hdf5` (lazy, if not already open)
   - Read `betas[0]` (first trial in session 3)
   - Shape: (num_voxels,)

4. **Lookup image ID**:
   ```python
   stim_id = self.trial_idx[1500]  # e.g., returns 5432
   ```

5. **Load image**:
   - Open `nsd_stimuli.hdf5` (lazy, if not already open)
   - Read `imgBrick[5432]`
   - Shape: (425, 425, 3)

6. **Return**:
   ```python
   {
       "voxels": tensor([...]),  # fMRI data
       "images": tensor([...]),  # Image data
       "trial": 1500             # Global trial number
   }
   ```

---

## Key Design Patterns

1. **Lazy Loading**: Files opened only when needed, not all at once
2. **Caching**: Open file handles stored to avoid repeated opens
3. **Index Mapping**: Three-level indexing system for efficient data access
4. **Separation of Concerns**: fMRI data in session files, images in one large file
5. **Memory Efficiency**: Only loads data for requested items, not entire dataset

