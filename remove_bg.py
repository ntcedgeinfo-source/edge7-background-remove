from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image, ImageOps
from rembg import remove, new_session

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp"}

MODEL_NAME = "u2netp"   # much faster than birefnet-portrait
PADDING = 30
SQUARE_SIZE = None
MAX_INPUT_SIZE = 1600   # resize very large images before background removal
WORKERS = 4             # tune based on CPU/RAM

session = new_session(MODEL_NAME)


def preprocess_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # resize huge images for faster inference
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_INPUT_SIZE:
        scale = MAX_INPUT_SIZE / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def crop_to_subject_rgba(img: Image.Image, padding: int = 0) -> Image.Image:
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
    img = img.convert("RGBA")
    w, h = img.size
    side = max(w, h) if size is None else size

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))

    if size is not None:
        scale = min(side / w, side / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size

    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def process_file(file: Path):
    try:
        output_file = OUTPUT_DIR / f"{file.stem}_portrait_square.png"

        with Image.open(file) as img:
            img = preprocess_image(img)

            result = remove(
                img,
                session=session,
                alpha_matting=False,   # fastest major change
                post_process_mask=True,
            )

            if isinstance(result, Image.Image):
                out = result
            else:
                out = Image.open(BytesIO(result)).convert("RGBA")

            out = crop_to_subject_rgba(out, padding=PADDING)
            out = make_square_canvas(out, size=SQUARE_SIZE)
            out.save(output_file, "PNG", optimize=True)

        return f"Processed: {file.name} -> {output_file.name}"
    except Exception as e:
        return f"Failed: {file.name} -> {e}"


files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in VALID_EXT]

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(process_file, f) for f in files]
    for future in as_completed(futures):
        print(future.result())
