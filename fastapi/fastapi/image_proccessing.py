from utils import *
from detection_model import *
from embedding_model import*
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw

def get_prediction(image: Image):
    detection_model = YoloStamp.from_pretrained("stamps-labs/yolo-stamp")

    image = image.convert("RGB")
    w, h = image.size

    image = np.array(image)
    transformer = A.Compose([
                A.Resize(height=448, width=448),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ])
    image = transformer(image=image)["image"]
    coef_w, coef_h = w/448, h/448
    output = detection_model(image.unsqueeze(0))

    prediction = output_tensor_to_boxes(output[0].detach().cpu())
    prediction = nonmax_suppression(prediction)
    
    for pred in prediction:
        pred[0] *= coef_w
        pred[1] *= coef_h
        pred[2] *= coef_w
        pred[3] *= coef_h
        
    return prediction

def get_prediction_response(prediction):
    res = []
    template = ["xmin", "ymin", "width", "height", "certainty"]
    for box in prediction:
        arr = []
        for i, tensor in enumerate(box):
            reshaped_tensor = tensor.view(1)
            numpy_array = reshaped_tensor.detach().numpy()
            n = numpy_array.tolist()[0]
            if i != 4:
                n = int(n)
            arr.append(n)
        res.append(dict(zip(template, arr)))
    return res

def get_boxes_coords(prediction):
    res = []
    for box in prediction:
        arr = []
        for i, tensor in enumerate(box):
            reshaped_tensor = tensor.view(1)
            numpy_array = reshaped_tensor.detach().numpy()
            n = numpy_array.tolist()[0]
            if i != 4:
                n = int(n)
                arr.append(n)
        res.append(arr)
    return res

def generate_image(image: Image, prediction):
    img = image.copy()
    drawer = ImageDraw.Draw(img)
    for box in prediction:
            box = (box[0], box[1]), (box[0] + box[2], box[1] + box[3])
            drawer.rectangle(box, outline = 'red')
    return img

def get_embed_predict(image: Image):
    model = EmbeddingModel()
    p = model.predict(image)
    return p