from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from image_proccessing import *
from io import BytesIO
import torch
app = FastAPI()

@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

@app.post("/image-w-boxes")
async def get_image(file: UploadFile):
    request_object_content = await file.read()
    image = Image.open(BytesIO(request_object_content)).convert("RGB")
    processed_image = generate_image(image, get_prediction(image))
    processed_image_bytes = BytesIO()
    processed_image.save(processed_image_bytes, format="JPEG")
    processed_image_bytes.seek(0)
    return StreamingResponse(content = processed_image_bytes, media_type="image/jpeg")

@app.post("/bounding-boxes")
async def get_bboxes(file: UploadFile):
    request_object_content = await file.read()
    image = Image.open(BytesIO(request_object_content))
    prediction = get_prediction(image)
    return {"bboxes" : get_prediction_response(prediction)}

@app.post("/embeddings-from-cropped")
async def get_embedding(file: UploadFile):
    request_object_content = await file.read()
    image = Image.open(BytesIO(request_object_content))
    p = get_embed_predict(image)
    p = p.tolist()
    return {"emb" : p}