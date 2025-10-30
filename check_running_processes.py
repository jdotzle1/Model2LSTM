#!/usr/bin/env python3
"""
Check Running Processes

This script helps identify if there are any hanging Python processes
and what they might be doing.
"""

import psutil
import time

def check_python_processes():
    """Check all running Python processes"""
    print("🔍 CHECKING RUNNING PYTHON PROCESSES")
    print("=" * 60)
    
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not python_processes:
        print("No Python processes found")
        return
    
    print(f"Found {len(python_processes)} Python processes:")
    print()
    
    for i, proc in enumerate(python_processes, 1):
        print(f"Process {i}:")
        print(f"  PID: {proc['pid']}")
        print(f"  Name: {proc['name']}")
        print(f"  CPU: {proc['cpu_percent']:.1f}%")
        print(f"  Memory: {proc['memory_percent']:.1f}%")
        
        # Show command line if available
        if proc['cmdline']:
            cmdline = ' '.join(proc['cmdline'])
            if len(cmdline) > 100:
                cmdline = cmdline[:100] + "..."
            print(f"  Command: {cmdline}")
        
        # Show how long it's been running
        create_time = proc['create_time']
        running_time = time.time() - create_time
        if running_time > 3600:
            print(f"  Running: {running_time/3600:.1f} hours")
        elif running_time > 60:
            print(f"  Running: {running_time/60:.1f} minutes")
        else:
            print(f"  Running: {running_time:.1f} seconds")
        
        print()

def check_nohup_processes():
    """Check for nohup processes specifically"""
    print("🔍 CHECKING NOHUP PROCESSES")
    print("=" * 60)
    
    nohup_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            if proc.info['cmdline'] and any('nohup' in str(cmd) for cmd in proc.info['cmdline']):
                nohup_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not nohup_processes:
        print("No nohup processes found")
        return
    
    print(f"Found {len(nohup_processes)} nohup processes:")
    print()
    
    for i, proc in enumerate(nohup_processes, 1):
        print(f"Nohup Process {i}:")
        print(f"  PID: {proc['pid']}")
        print(f"  CPU: {proc['cpu_percent']:.1f}%")
        print(f"  Memory: {proc['memory_percent']:.1f}%")
        
        if proc['cmdline']:
            cmdline = ' '.join(proc['cmdline'])
            print(f"  Command: {cmdline}")
        print()

def check_system_load():
    """Check overall system load"""
    print("🔍 SYSTEM LOAD CHECK")
    print("=" * 60)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent:.1f}% ({memory.used/(1024**3):.1f} GB / {memory.total/(1024**3):.1f} GB)")
    
    # Load average (Linux/Mac only)
    try:
        load_avg = psutil.getloadavg()
        print(f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    except AttributeError:
        print("Load average not available on this system")
    
    # Disk I/O
    try:
        disk_io = psutil.disk_io_counters()
        print(f"Disk Read: {disk_io.read_bytes/(1024**2):.1f} MB")
        print(f"Disk Write: {disk_io.write_bytes/(1024**2):.1f} MB")
    except AttributeError:
        print("Disk I/O stats not available")
    
    print()

def main():
    """Run all checks"""
    print("🚀 PROCESS DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    check_system_load()
    check_python_processes()
    check_nohup_processes()
    
    print("🔧 TROUBLESHOOTING TIPS:")
    print("1. If you see a Python process using high CPU/memory, it might still be working")
    print("2. If you see a process that's been running for hours with 0% CPU, it might be stuck")
    print("3. You can kill stuck processes with: kill -9 <PID>")
    print("4. Check nohup.out file for any error messages")
    print("5. Consider using 'screen' or 'tmux' instead of nohup for better control")

if __name__ == "__main__":
    main()