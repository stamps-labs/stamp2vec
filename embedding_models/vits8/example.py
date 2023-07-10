from PIL import Image
from model import ViTStamp
def get_embeddings(img_path: str):
    model = ViTStamp()
    image = Image.open(img_path)
    embeddings = model(image=image)
    return embeddings

if __name__ == "__main__":
    print(get_embeddings("oml/data/test/images/99d_15.bmp"))