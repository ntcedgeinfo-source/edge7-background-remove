from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from rembg import remove, new_session

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp"}

# Pick the best model for your case:
# "isnet-general-use" -> good general choice
# "u2net_human_seg"   -> better for people
# "isnet-anime"       -> anime
# "birefnet-portrait" -> portraits
MODEL_NAME = "birefnet-general"

session = new_session(MODEL_NAME)

def preprocess_image(img: Image.Image) -> Image.Image:
    # Fix phone photo rotation from EXIF
    img = ImageOps.exif_transpose(img)

    # Convert to RGB before processing
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Mild cleanup only. Do not over-sharpen.
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

for file in INPUT_DIR.iterdir():
    if file.suffix.lower() not in VALID_EXT:
        continue

    output_file = OUTPUT_DIR / f"{file.stem}_no_bg.png"

    with Image.open(file) as img:
        img = preprocess_image(img)

        result = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=8,
            post_process_mask=True,
        )

        if isinstance(result, Image.Image):
            out = result
        else:
            out = Image.open(BytesIO(result))

        out.save(output_file, "PNG")

    print(f"Processed: {file.name} -> {output_file.name}")
