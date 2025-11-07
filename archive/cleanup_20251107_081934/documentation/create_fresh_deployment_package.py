#!/usr/bin/env python3
"""
Create Fresh Deployment Package

This script creates a new deployment package with the CURRENT code,
ensuring EC2 runs the same code you're developing locally.
"""

import os
import shutil
import tarfile
from pathlib import Path
from datetime import datetime
import hashlib

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    print("=" * 80)
    print("CREATE FRESH DEPLOYMENT PACKAGE")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create deployment directory
    deploy_dir = project_root / 'deployment' / 'ec2'
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    package_name = f'ec2_deployment_package_{timestamp}'
    package_dir = deploy_dir / package_name
    
    print(f"\n📦 Creating package: {package_name}")
    
    # Create package structure
    package_src = package_dir / 'src'
    package_src.mkdir(parents=True, exist_ok=True)
    
    # Copy source code
    print("\n📋 Copying source code...")
    
    files_to_copy = {
        'src/data_pipeline': package_src / 'data_pipeline',
        'src/config': package_src / 'config',
        'src/__init__.py': package_src / '__init__.py',
    }
    
    for source, dest in files_to_copy.items():
        source_path = project_root / source
        
        if source_path.is_dir():
            print(f"  Copying directory: {source}")
            shutil.copytree(source_path, dest, dirs_exist_ok=True)
        elif source_path.is_file():
            print(f"  Copying file: {source}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest)
    
    # Copy main processing script
    print("\n📋 Copying processing scripts...")
    shutil.copy2(
        project_root / 'process_monthly_chunks_fixed.py',
        package_dir / 'process_monthly_chunks_fixed.py'
    )
    
    # Copy requirements
    print("\n📋 Copying requirements...")
    shutil.copy2(
        project_root / 'requirements.txt',
        package_dir / 'requirements.txt'
    )
    
    # Create README
    print("\n📋 Creating deployment README...")
    readme_content = f"""# EC2 Deployment Package - {timestamp}

## Package Contents

- `src/` - Production source code
- `process_monthly_chunks_fixed.py` - Main processing script
- `requirements.txt` - Python dependencies

## Deployment Instructions

1. Upload this package to EC2:
   ```bash
   scp ec2_deployment_package_{timestamp}.tar.gz ec2-user@your-instance:/home/ec2-user/
   ```

2. On EC2, extract the package:
   ```bash
   tar -xzf ec2_deployment_package_{timestamp}.tar.gz
   cd ec2_deployment_package_{timestamp}
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify the code:
   ```bash
   md5sum src/data_pipeline/weighted_labeling.py
   # Should match local hash
   ```

5. Run processing:
   ```bash
   python process_monthly_chunks_fixed.py --month 2011-06
   ```

## File Hashes

### Critical Files:
"""
    
    # Calculate hashes for critical files
    critical_files = [
        'src/data_pipeline/weighted_labeling.py',
        'src/data_pipeline/features.py',
        'src/data_pipeline/pipeline.py',
        'process_monthly_chunks_fixed.py',
    ]
    
    hashes = {}
    for file_path in critical_files:
        full_path = package_dir / file_path
        if full_path.exists():
            file_hash = calculate_md5(full_path)
            hashes[file_path] = file_hash
            readme_content += f"\n- `{file_path}`: {file_hash}"
    
    readme_content += f"""

## Verification

After deployment, verify these hashes match on EC2:

```bash
md5sum src/data_pipeline/weighted_labeling.py
md5sum src/data_pipeline/features.py
md5sum src/data_pipeline/pipeline.py
md5sum process_monthly_chunks_fixed.py
```

## Package Created

- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Local machine: {os.uname().nodename if hasattr(os, 'uname') else 'Windows'}
- Python version: {os.sys.version.split()[0]}
"""
    
    with open(package_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # Create hash manifest
    print("\n📋 Creating hash manifest...")
    manifest_path = package_dir / 'MANIFEST.txt'
    with open(manifest_path, 'w') as f:
        f.write(f"Deployment Package Manifest - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        for file_path, file_hash in hashes.items():
            f.write(f"{file_hash}  {file_path}\n")
    
    # Create tarball
    print("\n📦 Creating tarball...")
    tarball_path = deploy_dir / f'{package_name}.tar.gz'
    
    with tarfile.open(tarball_path, 'w:gz') as tar:
        tar.add(package_dir, arcname=package_name)
    
    tarball_size = tarball_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Package created: {tarball_path}")
    print(f"   Size: {tarball_size:.2f} MB")
    
    # Show comparison with old package
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    print("\n📋 File Hashes:")
    for file_path, file_hash in hashes.items():
        print(f"  {file_path}")
        print(f"    NEW: {file_hash}")
        
        # Check old package if it exists
        old_package_path = deploy_dir / 'ec2_deployment_package' / 'project' / 'project' / file_path.replace('src/', '')
        if old_package_path.exists():
            old_hash = calculate_md5(old_package_path)
            if old_hash == file_hash:
                print(f"    OLD: {old_hash} ✅ SAME")
            else:
                print(f"    OLD: {old_hash} ❌ DIFFERENT")
        else:
            print(f"    OLD: Not found")
    
    # Create deployment instructions
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    print(f"""
1. Upload to EC2:
   scp {tarball_path} ec2-user@your-instance:/home/ec2-user/

2. On EC2, extract and verify:
   tar -xzf {package_name}.tar.gz
   cd {package_name}
   cat MANIFEST.txt
   md5sum src/data_pipeline/weighted_labeling.py

3. Install dependencies:
   pip install -r requirements.txt

4. Run processing:
   python process_monthly_chunks_fixed.py --month 2011-06

5. Compare results to old run:
   - Old code: 66% short win rates
   - New code: ??? win rates
""")
    
    print("\n✅ Deployment package ready!")

if __name__ == "__main__":
    main()
