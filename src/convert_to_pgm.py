from PIL import Image
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_to_pgm.py input.jpg output.pgm")
    exit(1)

img = Image.open(sys.argv[1]).convert("L")  # Convert to grayscale
img.save(sys.argv[2])
