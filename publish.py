#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

def run(cmd):
    if subprocess.run(cmd, shell=isinstance(cmd, str)).returncode != 0:
        sys.exit(1)

def update_version(setup_file, new_version):
    content = setup_file.read_text()
    content = re.sub(
        r'version\s*=\s*["\'][^"\']+["\']',
        f'version="{new_version}"',
        content
    )
    setup_file.write_text(content)
    print(f"✓ Updated version to {new_version}")

def clean():
    for path in ["dist", "build", "*.egg-info"]:
        for item in Path(".").glob(path):
            shutil.rmtree(item, ignore_errors=True)
    print("✓ Cleaned build artifacts")

def main():
    if len(sys.argv) < 2:
        print("Usage: python publish.py <version> [--test]")
        print("Example: python publish.py 1.0.5")
        sys.exit(1)
    
    version = sys.argv[1]
    use_test = "--test" in sys.argv
    
    setup_file = Path("setup.py")
    if not setup_file.exists():
        print("✗ setup.py not found")
        sys.exit(1)
    
    token_var = "TESTPYPI_API_TOKEN" if use_test else "PYPI_API_TOKEN"
    token = os.environ.get(token_var)
    if not token:
        print(f"✗ {token_var} environment variable not set")
        sys.exit(1)
    
    update_version(setup_file, version)
    
    clean()
    print("Building package...")
    run([sys.executable, "-m", "build"])
    print("✓ Built package")
    
    repo_url = "https://test.pypi.org/legacy/" if use_test else "https://upload.pypi.org/legacy/"
    print(f"Uploading to {'TestPyPI' if use_test else 'PyPI'}...")
    run([
        sys.executable, "-m", "twine", "upload",
        "--repository-url", repo_url,
        "--username", "__token__",
        "--password", token,
        "dist/*"
    ])
    
    print(f"\n✓ Published version {version}")
    if use_test:
        print(f"  https://test.pypi.org/project/lib-memvid/{version}/")
    else:
        print(f"  https://pypi.org/project/lib-memvid/{version}/")

if __name__ == "__main__":
    main()
