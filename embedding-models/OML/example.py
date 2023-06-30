from PIL import Image
from inference import EmbeddingModel

def test_infer():
    model = EmbeddingModel()
    image = Image.open("data/test/images/100a_0.png")
    embeddings = model.predict(image=image)
    print(embeddings)


if __name__ == "__main__":
    test_infer()