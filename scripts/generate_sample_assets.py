import os
from PIL import Image, ImageDraw

def generate_ad_file(filename, text, color1, color2, is_bottle=False):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "ad_images")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, filename)

    if os.path.exists(out_path):
        return

    # 400x600 for bottle, 400x300 for banner
    w, h = (400, 600) if is_bottle else (400, 300)
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(h):
        r = int(color1[0] + (y/h) * (color2[0] - color1[0]))
        g = int(color1[1] + (y/h) * (color2[1] - color1[1]))
        b = int(color1[2] + (y/h) * (color2[2] - color1[2]))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    if is_bottle:
        # Draw a simple amber oil bottle shape
        draw.ellipse([100, 150, 300, 550], fill=(210, 105, 30), outline=(139, 69, 19), width=5)
        draw.rectangle([160, 50, 240, 150], fill=(139, 69, 19))
        draw.text((115, 300), "PREMIUM OIL", fill=(255, 255, 255))
    else:
        # Draw standard banner text
        draw.text((120, 130), text, fill=(0, 0, 0))

    img.save(out_path)
    print(f"Sample asset created at {out_path}")

def generate_sample_ad():
    # 1. Standard Sample Ad
    generate_ad_file("sample_ad.png", "SAMPLE AD", (255, 200, 150), (200, 100, 50))
    # 2. Oil Ad (for README example)
    generate_ad_file("oil_ad.png", "PREMIUM OIL", (255, 255, 200), (200, 180, 50), is_bottle=True)

if __name__ == "__main__":
    generate_sample_ad()
