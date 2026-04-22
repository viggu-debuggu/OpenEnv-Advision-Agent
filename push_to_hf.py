import subprocess
import os

# 1. Configuration
GITHUB_URL = "https://github.com/viggu-debuggu/OpenEnv-Advision-Agent.git"
HF_SPACE_URL = "https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent"

def run(cmd):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 2. Refined .gitignore for OpenEnv Compliance
gitignore = """
.env
.venv
advision_env/__pycache__/
server/__pycache__/
__pycache__/
*.pyc
.ruff_cache
.ipynb_checkpoints
data/output_videos/*
server/static/temp/*
# Keep samples for UI functionality
!data/input_videos/test.mp4
!data/ad_images/oil_ad.png
!data/ad_images/sample_ad.png
!server/static/assets/*
# Keep models
!yolov8n.pt
Thumbs.db
.DS_Store
"""

with open('.gitignore', 'w') as f:
    f.write(gitignore)

try:
    print("Initializing AdVision Deployment...")
    
    # Check if git is initialized
    if not os.path.exists(".git"):
        run("git init")
    
    run("git add .")
    run('git commit -m "Finalized AdVision: Stunning UI + OpenEnv v1.0 Compliance"')

    # 3. Handle Remotes
    print("\nConfiguring Remotes...")
    try:
        run(f"git remote add github {GITHUB_URL}")
    except:
        run(f"git remote set-url github {GITHUB_URL}")
        
    try:
        run(f"git remote add hf {HF_SPACE_URL}")
    except:
        run(f"git remote set-url hf {HF_SPACE_URL}")

    # 4. Push to repositories
    print("\nPushing to GitHub...")
    run("git push -u github main --force")
    
    print("\nPushing to Hugging Face Spaces...")
    run("git push -u hf main --force")
    
    print("\nAdVision successfully deployed to GitHub and Hugging Face!")
    print(f"GitHub: {GITHUB_URL}")
    print(f"Space:  {HF_SPACE_URL}")

except Exception as e:
    print(f"\nError during deployment: {e}")
    print("Hint: Ensure you have 'git' installed and are logged in to GitHub/HF.")
