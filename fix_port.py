#!/usr/bin/env python3
"""
Script to fix port conflicts by finding and killing processes on specific ports.
"""

import subprocess
import platform
import sys
import argparse

def find_processes_on_port(port: int) -> list:
    """Find processes running on a specific port."""
    processes = []
    system = platform.system().lower()
    
    try:
        if system == "linux" or system == "darwin":  # Linux or macOS
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        processes.append(pid.strip())
        elif system == "windows":
            result = subprocess.run(['netstat', '-ano'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            processes.append(parts[-1])
    except Exception as e:
        print(f"Error finding processes: {e}")
    
    return processes

def kill_processes_on_port(port: int, force: bool = False) -> bool:
    """Kill processes running on a specific port."""
    processes = find_processes_on_port(port)
    
    if not processes:
        print(f"No processes found on port {port}")
        return True
    
    print(f"Found {len(processes)} process(es) on port {port}:")
    for pid in processes:
        print(f"  - PID: {pid}")
    
    if not force:
        response = input(f"Do you want to kill these processes? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return False
    
    system = platform.system().lower()
    killed_count = 0
    
    for pid in processes:
        try:
            if system == "linux" or system == "darwin":
                subprocess.run(['kill', '-9', pid], 
                             capture_output=True, check=True)
            elif system == "windows":
                subprocess.run(['taskkill', '/PID', pid, '/F'], 
                             capture_output=True, check=True)
            print(f"✅ Killed process {pid}")
            killed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to kill process {pid}: {e}")
    
    print(f"Killed {killed_count}/{len(processes)} processes")
    return killed_count > 0

def main():
    parser = argparse.ArgumentParser(description='Fix port conflicts by killing processes')
    parser.add_argument('port', type=int, help='Port number to check')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Force kill without confirmation')
    parser.add_argument('--list', '-l', action='store_true',
                       help='Only list processes, do not kill')
    
    args = parser.parse_args()
    
    print(f"🔍 Checking port {args.port}...")
    
    if args.list:
        processes = find_processes_on_port(args.port)
        if processes:
            print(f"Found {len(processes)} process(es) on port {args.port}:")
            for pid in processes:
                print(f"  - PID: {pid}")
        else:
            print(f"No processes found on port {args.port}")
    else:
        if kill_processes_on_port(args.port, args.force):
            print(f"✅ Successfully cleared port {args.port}")
        else:
            print(f"❌ Failed to clear port {args.port}")
            sys.exit(1)

if __name__ == "__main__":
    main() 