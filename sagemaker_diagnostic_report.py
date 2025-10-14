import os
import subprocess
import sys
from pathlib import Path

print("\n" + "="*60)
print("SAGEMAKER DIAGNOSTIC REPORT")
print("="*60 + "\n")

# 1. Check git repository
print("1. GIT REPOSITORY STATUS")
print("-" * 40)
try:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print(f"✅ Git repository: YES")
    else:
        print(f"❌ Git repository: NO")
except Exception as e:
    print(f"❌ Git check failed: {e}")

# 2. Check .gitignore exists
print("\n2. .GITIGNORE FILE")
print("-" * 40)
if os.path.exists(".gitignore"):
    print("✅ .gitignore exists")
    with open(".gitignore", "r") as f:
        lines = f.readlines()
    print(f"   Lines: {len(lines)}")
    print("   Content (first 10 lines):")
    for line in lines[:10]:
        print(f"   {line.rstrip()}")
else:
    print("❌ .gitignore does NOT exist")

# 3. Check problematic directories
print("\n3. DIRECTORY SIZES")
print("-" * 40)
problematic_dirs = [
    "launcher_venv",
    "sagemaker_env",
    "venv",
    ".venv",
    "data",
    "outputs",
    "__pycache__",
]

for dir_name in problematic_dirs:
    if os.path.exists(dir_name):
        try:
            result = subprocess.run(
                ["du", "-sh", dir_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            size = result.stdout.split()[0]
            print(f"   {dir_name:20s} {size}")
        except Exception as e:
            print(f"   {dir_name:20s} (error: {e})")
    else:
        print(f"   {dir_name:20s} (not found)")

# 4. Estimate what will be archived
print("\n4. ARCHIVE ESTIMATION (simulating git archive behavior)")
print("-" * 40)
try:
    # This simulates what SageMaker does
    result = subprocess.run(
        ["git", "ls-files", "-o", "-i", "--exclude-standard"],
        capture_output=True,
        text=True,
        timeout=10
    )
    ignored_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    print(f"   Ignored files/dirs: {len(ignored_files)}")
    if ignored_files and ignored_files[0]:
        print("   First few ignored items:")
        for item in ignored_files[:5]:
            if item:
                print(f"      {item}")
except Exception as e:
    print(f"   Error checking ignored files: {e}")

# 5. Check what's tracked by git
print("\n5. GIT TRACKED FILES")
print("-" * 40)
try:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        timeout=10
    )
    tracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
    print(f"   Total tracked files: {len(tracked)}")
    print("   First 10 tracked files:")
    for f in tracked[:10]:
        if f:
            print(f"      {f}")
except Exception as e:
    print(f"   Error: {e}")

# 6. Check AWS credentials
print("\n6. AWS CREDENTIALS")
print("-" * 40)
if os.environ.get("AWS_ACCESS_KEY_ID"):
    print("✅ AWS_ACCESS_KEY_ID is set")
else:
    print("❌ AWS_ACCESS_KEY_ID is NOT set")

if os.environ.get("AWS_SECRET_ACCESS_KEY"):
    print("✅ AWS_SECRET_ACCESS_KEY is set")
else:
    print("❌ AWS_SECRET_ACCESS_KEY is NOT set")

if os.environ.get("AWS_DEFAULT_REGION"):
    print(f"✅ AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION')}")
else:
    print("⚠️  AWS_DEFAULT_REGION not set")

print("\n" + "="*60)
print("END OF DIAGNOSTIC REPORT")
print("="*60 + "\n")
