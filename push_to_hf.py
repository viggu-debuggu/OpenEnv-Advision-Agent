import subprocess

# 1. Initialize Git in the project root
def run(cmd):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

gitignore = """
.env
.venv
__pycache__
*.pyc
.ruff_cache
*.pt
*.mp4
*.png
*.jpg
.ipynb_checkpoints
data/input_videos/*
data/output_videos/*
data/ad_images/oil_ad.png
data/ad_images/sample_ad.png
# Exclude actual images used in dev, keep placeholders if needed.
Thumbs.db
"""

with open('.gitignore', 'w') as f:
    f.write(gitignore)

try:
    # 2. Add files and commit
    run("git init")
    run("git add .")
    run('git commit -m "OpenEnv Submission: Finalized AdVision Environment"')

    # 3. Add remotes
    # Replace these URLs with your actual repo URLs if needed
    # run("git remote add origin https://github.com/vignesh-debuggu/AdVision.git")
    # run("git remote add hf https://huggingface.co/spaces/vignesh-debuggu/AdVision")

    # 4. Push to repositories
    # run("git push -u origin main --force")
    # run("git push -u hf main --force")
    print("\n✅ Successfully initialized and committed local changes.")
    print("   To push, uncomment the git remote and push lines above.")

except Exception as e:
    print(f"❌ Error during git process: {e}")
