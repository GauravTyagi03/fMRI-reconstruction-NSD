#!/usr/bin/env python3
"""
Memory monitoring script for training.
Run this in a separate terminal while training to monitor memory usage.
"""

import psutil
import time
import os
import subprocess
import sys

def get_cpu_memory():
    """Get CPU RAM usage"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024**3,  # GB
        'used': mem.used / 1024**3,
        'available': mem.available / 1024**3,
        'percent': mem.percent
    }

def get_gpu_memory():
    """Get GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            used, total = map(int, line.split(', '))
            gpus.append({
                'used': used / 1024,  # Convert MB to GB
                'total': total / 1024,
                'percent': (used / total) * 100
            })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_process_memory(process_name='Train_MindEye'):
    """Get memory usage of specific process"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                mem_mb = proc.info['memory_info'].rss / 1024**2
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': mem_mb,
                    'memory_gb': mem_mb / 1024
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def print_memory_stats():
    """Print formatted memory statistics"""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 60)
    print("MEMORY MONITORING")
    print("=" * 60)
    print()
    
    # CPU Memory
    cpu_mem = get_cpu_memory()
    print("CPU RAM (System Memory):")
    print(f"  Total:    {cpu_mem['total']:.2f} GB")
    print(f"  Used:     {cpu_mem['used']:.2f} GB ({cpu_mem['percent']:.1f}%)")
    print(f"  Available: {cpu_mem['available']:.2f} GB")
    print()
    
    # GPU Memory
    gpu_mem = get_gpu_memory()
    if gpu_mem:
        print("GPU Memory (VRAM):")
        for i, gpu in enumerate(gpu_mem):
            print(f"  GPU {i}:")
            print(f"    Used:     {gpu['used']:.2f} GB ({gpu['percent']:.1f}%)")
            print(f"    Total:    {gpu['total']:.2f} GB")
            print(f"    Available: {gpu['total'] - gpu['used']:.2f} GB")
    else:
        print("GPU Memory: nvidia-smi not available")
    print()
    
    # Process Memory
    processes = get_process_memory()
    if processes:
        print("Training Process Memory:")
        total_mem = 0
        for proc in processes:
            print(f"  PID {proc['pid']} ({proc['name']}): {proc['memory_gb']:.2f} GB")
            total_mem += proc['memory_gb']
        print(f"  Total: {total_mem:.2f} GB")
    else:
        print("Training Process: Not found")
    print()
    
    # Warnings
    print("Warnings:")
    warnings = []
    if cpu_mem['percent'] > 80:
        warnings.append(f"⚠️  High CPU RAM usage: {cpu_mem['percent']:.1f}%")
    if gpu_mem:
        for i, gpu in enumerate(gpu_mem):
            if gpu['percent'] > 90:
                warnings.append(f"⚠️  High GPU {i} memory: {gpu['percent']:.1f}%")
    if not warnings:
        print("  ✅ All systems normal")
    else:
        for warning in warnings:
            print(f"  {warning}")
    
    print()
    print("=" * 60)
    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to exit")

def main():
    """Main monitoring loop"""
    try:
        interval = 2  # Update every 2 seconds
        if len(sys.argv) > 1:
            interval = float(sys.argv[1])
        
        print("Starting memory monitor (update interval: {}s)".format(interval))
        print("Press Ctrl+C to exit\n")
        time.sleep(1)
        
        while True:
            print_memory_stats()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()

