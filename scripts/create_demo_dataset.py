from pathlib import Path
import random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "data" / "dataset"
random.seed(42)

for label in ["normal", "defective"]:
    (ROOT / label).mkdir(parents=True, exist_ok=True)

for index in range(40):
    image = Image.new("RGB", (256, 256), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, 220, 220), radius=18, fill=(150, 160, 170), outline=(80, 90, 100), width=4)
    draw.ellipse((72, 72, 184, 184), fill=(195, 200, 205), outline=(120, 125, 130), width=3)
    image.save(ROOT / "normal" / f"normal_{index:03d}.png")

for index in range(40):
    image = Image.new("RGB", (256, 256), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, 220, 220), radius=18, fill=(150, 160, 170), outline=(80, 90, 100), width=4)
    draw.ellipse((72, 72, 184, 184), fill=(195, 200, 205), outline=(120, 125, 130), width=3)
    start = random.randint(68, 150)
    draw.line((start, 58, start + 35, 185), fill=(205, 50, 50), width=7)
    image.save(ROOT / "defective" / f"defective_{index:03d}.png")

print(ROOT)
