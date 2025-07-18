from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("./output/result.pgm")
plt.imshow(img, cmap='gray')
plt.title("PGM Image")
plt.axis('off')
plt.show()
