from PIL import Image
from model_oml import EmbeddingModelOML

def get_embeddings(img_path: str):
    model = EmbeddingModelOML()
    image = Image.open(img_path)
    embeddings = model(image=image)
    return embeddings


if __name__ == "__main__":
    print(get_embeddings("data/test/images/99d_15.bmp"))