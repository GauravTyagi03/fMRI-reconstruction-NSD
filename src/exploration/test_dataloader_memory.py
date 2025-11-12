#!/usr/bin/env python3
"""
Test script to verify dataloader memory usage before full training.
This helps ensure your 64GB RAM is sufficient.
"""

import torch
import psutil
import os
import sys
import argparse
from dataloader import get_dataloaders

def format_bytes(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    return {
        'process_rss': mem_info.rss,  # Resident Set Size
        'process_vms': mem_info.vms,   # Virtual Memory Size
        'system_total': system_mem.total,
        'system_used': system_mem.used,
        'system_available': system_mem.available,
        'system_percent': system_mem.percent
    }

def test_dataloader_memory(fmri_dir, stimulus_path, train_idx_path, val_idx_path,
                           batch_size=32, num_workers=4, num_batches=20):
    """Test dataloader memory usage"""
    
    print("=" * 70)
    print("DATALOADER MEMORY TEST")
    print("=" * 70)
    print()
    
    # Initial memory
    mem_before = get_memory_info()
    print("Initial Memory State:")
    print(f"  Process RSS: {format_bytes(mem_before['process_rss'])}")
    print(f"  System RAM: {format_bytes(mem_before['system_used'])} / {format_bytes(mem_before['system_total'])} ({mem_before['system_percent']:.1f}%)")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    try:
        train_dl, val_dl, num_train, num_val = get_dataloaders(
            batch_size=batch_size,
            fmri_dir=fmri_dir,
            stimulus_path=stimulus_path,
            train_idx_path=train_idx_path,
            val_idx_path=val_idx_path,
            num_workers=num_workers
        )
        print(f"  ✓ Created dataloaders")
        print(f"  Training samples: {num_train}")
        print(f"  Validation samples: {num_val}")
        print()
    except Exception as e:
        print(f"  ✗ Error creating dataloaders: {e}")
        return False
    
    # Memory after creating dataloaders
    mem_after_init = get_memory_info()
    init_delta = mem_after_init['process_rss'] - mem_before['process_rss']
    print("After Creating Dataloaders:")
    print(f"  Process RSS: {format_bytes(mem_after_init['process_rss'])} (Δ {format_bytes(init_delta)})")
    print(f"  System RAM: {format_bytes(mem_after_init['system_used'])} ({mem_after_init['system_percent']:.1f}%)")
    print()
    
    # Test loading batches
    print(f"Loading {num_batches} batches...")
    print()
    
    max_mem = mem_after_init['process_rss']
    max_system = mem_after_init['system_used']
    
    try:
        for i, batch in enumerate(train_dl):
            mem_current = get_memory_info()
            max_mem = max(max_mem, mem_current['process_rss'])
            max_system = max(max_system, mem_current['system_used'])
            
            if (i + 1) % 5 == 0 or i == 0:
                batch_delta = mem_current['process_rss'] - mem_after_init['process_rss']
                print(f"  Batch {i+1:3d}: Process RSS = {format_bytes(mem_current['process_rss'])} "
                      f"(Δ {format_bytes(batch_delta)}), "
                      f"System RAM = {format_bytes(mem_current['system_used'])} ({mem_current['system_percent']:.1f}%)")
            
            if i >= num_batches - 1:
                break
            
            # Explicitly delete batch to test garbage collection
            del batch
            
    except Exception as e:
        print(f"  ✗ Error loading batches: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final memory
    mem_final = get_memory_info()
    print()
    print("Final Memory State:")
    print(f"  Process RSS: {format_bytes(mem_final['process_rss'])}")
    print(f"  Peak Process RSS: {format_bytes(max_mem)}")
    print(f"  System RAM: {format_bytes(mem_final['system_used'])} ({mem_final['system_percent']:.1f}%)")
    print(f"  Peak System RAM: {format_bytes(max_system)}")
    print()
    
    # Summary
    total_delta = mem_final['process_rss'] - mem_before['process_rss']
    peak_delta = max_mem - mem_before['process_rss']
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Memory increase from dataloader: {format_bytes(total_delta)}")
    print(f"Peak memory increase: {format_bytes(peak_delta)}")
    print(f"Current system RAM usage: {mem_final['system_percent']:.1f}%")
    print()
    
    # Recommendations
    if mem_final['system_percent'] > 80:
        print("⚠️  WARNING: System RAM usage is high (>80%)")
        print("   Consider reducing num_workers or batch_size")
    elif mem_final['system_percent'] > 60:
        print("⚠️  CAUTION: System RAM usage is moderate (>60%)")
        print("   Monitor during full training")
    else:
        print("✅ System RAM usage is healthy (<60%)")
        print("   You should be fine for full training")
    
    print()
    
    # Cleanup
    del train_dl, val_dl
    import gc
    gc.collect()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test dataloader memory usage')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NSD data directory')
    parser.add_argument('--subj', type=int, default=1,
                       help='Subject number')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--num_batches', type=int, default=20,
                       help='Number of batches to test')
    
    args = parser.parse_args()
    
    # Construct paths
    fmri_dir = os.path.join(args.data_path, "nsddata_betas", "ppdata", 
                           f"subj{args.subj}", "func1mm", "betas_fithrf_GLMdenoise_RR")
    stimulus_path = os.path.join(args.data_path, "nsddata_stimuli", "stimuli", 
                                 "nsd", "nsd_stimuli.hdf5")
    train_idx_path = os.path.join(args.data_path, "nsddata", 
                                  f"subj{args.subj}_train_idx.npy")
    val_idx_path = os.path.join(args.data_path, "nsddata", 
                                f"subj{args.subj}_val_idx.npy")
    
    # Verify paths exist
    for path, name in [(fmri_dir, "fMRI directory"), 
                       (stimulus_path, "Stimulus file"),
                       (train_idx_path, "Train indices"),
                       (val_idx_path, "Val indices")]:
        if not os.path.exists(path):
            print(f"✗ Error: {name} not found: {path}")
            sys.exit(1)
    
    # Run test
    success = test_dataloader_memory(
        fmri_dir=fmri_dir,
        stimulus_path=stimulus_path,
        train_idx_path=train_idx_path,
        val_idx_path=val_idx_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

