import numpy as np

from fastapi import FastAPI, Request
from nn.tasks import ConvModel
from nn.modules.utils import softmax
from utils import fetch_image_bytes_from_url, preprocess_to_tensor

app = FastAPI()
model = ConvModel()
model.load_params("data/params.pkl")

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    
    img_bytes = fetch_image_bytes_from_url(str(body["image_url"]))

    x = preprocess_to_tensor(img_bytes)

    logits = model.predict(x)
    probs = softmax(logits)
    answer = int(np.argmax(probs, axis=1)[0])

    return {"answer": answer}
