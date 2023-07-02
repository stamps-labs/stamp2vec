import ast
import os
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
import streamlit as st
import io

api = "http://127.0.0.1:8000/"

st.title("Stamp2vec")

input_image = st.file_uploader("insert image")
image = Image.open(input_image)
if(input_image):
    st.header("Original")
    st.image(input_image, use_column_width=True)
    if st.button("Get prediction"):
        col1, col2 = st.columns(2)
        col1.subheader("Prediction")
        response = requests.post(os.path.join(api, "bounding-boxes/"), files = {"file": input_image.getvalue()})
        prediction = ast.literal_eval(response.text)
        col1.write(prediction)

        col2.subheader("Image")
        response = requests.post(os.path.join(api, "image-w-boxes/"), files = {"file": input_image.getvalue()})
        col2.image(response.content, use_column_width=True)

        arr = []
        for b in prediction["bboxes"]:
            stamp = image.crop((b["xmin"], b["ymin"], b["xmin"] + b["width"], b["ymin"] + b["height"]))
            output = io.BytesIO()
            stamp.save(output, format="BMP")
            response = ast.literal_eval(requests.post(os.path.join(api, "embeddings-from-cropped/"), files = {"file": output.getvalue()}).text)
            arr.append(response["emb"])

        st.write(arr)
                
