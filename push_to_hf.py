import os, subprocess

# 1. Initialize Git in the project root
def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 2. Create .gitignore to keep the repo clean and under HF limits
gitignore = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
venv/
.env
# Ignore heavy output videos (keep repo size down)
data/output_videos/
# OS junk
.DS_Store
Thumbs.db
"""
with open('.gitignore', 'w') as f: f.write(gitignore)

try:
    run("git init")
    run("git add .")
    run("git commit -m 'Initial OpenEnv AdVision Submission'")
    
    # Target URL (the Space given by user)
    remote_url = "https://huggingface.co/spaces/vignesh93917/OpenEnv_AdVision_Agent"
    
    print("\n" + "="*50)
    print("READY TO PUSH!")
    print("="*50)
    print("Please run this command in your terminal to finish the link:")
    print(f"\ngit push --force {remote_url} main\n")
    print("Note: When prompted for a password, paste your Hugging Face WRITE token.")
    print("Get your token here: https://huggingface.co/settings/tokens")
    print("="*50)

except Exception as e:
    print(f"Git setup error: {e}")
