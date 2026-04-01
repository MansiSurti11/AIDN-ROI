import gdown
import os

url = 'https://drive.google.com/uc?id=1l0vsFlbiy3KkOM-dExOHmTQTWpJbo1wO'
output = 'LOG/DIV2K/pre-train/model_best.pth'

if not os.path.exists(output):
    print(f"Downloading weights to {output}...")
    gdown.download(url, output, quiet=False)
else:
    print("Weights already exist.")
