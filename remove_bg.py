from pathlib import Path
from rembg import remove

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

valid_ext = {".png", ".jpg", ".jpeg", ".webp"}

for file in INPUT_DIR.iterdir():
    if file.suffix.lower() not in valid_ext:
        continue

    output_file = OUTPUT_DIR / f"{file.stem}_no_bg.png"

    with open(file, "rb") as inp:
        data = inp.read()

    result = remove(data)

    with open(output_file, "wb") as out:
        out.write(result)

    print(f"Processed: {file.name} -> {output_file.name}")
