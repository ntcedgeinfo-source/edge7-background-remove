from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from rembg import remove, new_session

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp"}

MODEL_NAME = "birefnet-portrait"
PADDING = 30          # space around subject
SQUARE_SIZE = None    # None = keep natural square size, e.g. 1024 for fixed export

session = new_session(MODEL_NAME)


def preprocess_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def crop_to_subject_rgba(img: Image.Image, padding: int = 0) -> Image.Image:
    """
    Tight crop using alpha channel after background removal.
    """
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    return img.crop((left, top, right, bottom))

def make_square_canvas(img: Image.Image, size: int | None = None) -> Image.Image:
    """
    Center cropped subject on a transparent 1:1 canvas.
    If size is None, use the larger side of the cropped image.
    """
    img = img.convert("RGBA")
    w, h = img.size
    side = max(w, h) if size is None else size

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))

    if size is not None:
        # resize while keeping aspect ratio
        scale = min(side / w, side / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y), img)

    return canvas


for file in INPUT_DIR.iterdir():
    if file.suffix.lower() not in VALID_EXT:
        continue

    output_file = OUTPUT_DIR / f"{file.stem}_portrait_square.png"

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

        out = crop_to_subject_rgba(out, padding=PADDING)
        out = make_square_canvas(out, size=SQUARE_SIZE)
        out.save(output_file, "PNG")

    print(f"Processed: {file.name} -> {output_file.name}")
