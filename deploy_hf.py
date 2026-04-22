import os
import shutil
import subprocess
import tempfile

# Configuration
HF_SPACE_URL = "https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent"
SOURCE_DIR = os.getcwd()

# Files/Dirs to include (Source only, no binaries)
INCLUDE_PATHS = [
    "advision_env",
    "server",
    "Dockerfile",
    "requirements.txt",
    "entrypoint.sh",
    ".dockerignore",
    "pyproject.toml",
    "openenv.yaml",
    "inference.py",
    "README.md",
    "setup.py"
]

def run(cmd, cwd=None):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Deploying to HF via temp dir: {tmp_dir}")
        
        # 1. Copy files
        for path in INCLUDE_PATHS:
            src = os.path.join(SOURCE_DIR, path)
            dst = os.path.join(tmp_dir, path)
            if not os.path.exists(src):
                print(f"Skipping {path} (not found)")
                continue
            
            print(f"Copying {path}...")
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '*.png', '*.mp4', '*.pt'))
            elif os.path.isfile(src):
                shutil.copy2(src, dst)
        
        # 2. Initialize Git
        run("git init", cwd=tmp_dir)
        run("git checkout -b main", cwd=tmp_dir)
        run("git add .", cwd=tmp_dir)
        
        # Check if there are changes to commit
        status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True, cwd=tmp_dir)
        if not status.stdout.strip():
            print("Nothing to commit. Check if source files are present.")
        else:
            run('git commit -m "Finalized Premium UI Deployment"', cwd=tmp_dir)
            
            # 3. Add Remote
            run(f"git remote add origin {HF_SPACE_URL}", cwd=tmp_dir)
            
            # 4. Push Force
            print("Pushing to Hugging Face...")
            run("git push origin main --force", cwd=tmp_dir)
            
            print("\nSUCCESS: AdVision UI deployed to Hugging Face Space!")

except Exception as e:
    print(f"\nFAILED: {e}")
