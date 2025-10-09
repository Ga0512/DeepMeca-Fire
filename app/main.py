import os
import cv2
import bz2
import time
import json
import mimetypes
import requests
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# ===========================
# CONFIGURAÇÕES GERAIS
# ===========================

ACCESS_TOKEN = ""
PHONE_NUMBER_ID = ""
API_VERSION = "v22.0"

NGROK_TOKEN = ""
NGROK_PORT = 9192

# ===========================
# INICIALIZAÇÃO DE MODELOS
# ===========================

print("🔹 Carregando modelos...")

model_yolo = YOLO("fire_n.pt")

with bz2.BZ2File("regression.pkl", "rb") as f:
    model_reg = pickle.load(f)

scaler = StandardScaler()
df = pd.read_csv("Algerian_forest_fires_dataset_CLEANED.csv")
scaler.fit(df[["Temperature", "Ws", "FFMC", "DMC", "ISI"]])

print("✅ Modelos carregados com sucesso!")

# ===========================
# DEFINIÇÃO DA API FASTAPI
# ===========================

app = FastAPI(title="DeepMeca-Fire API", description="API para detecção de incêndios e previsão de FWI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# MODELOS Pydantic
# ===========================

class PredictRequest(BaseModel):
    Temperature: float
    Ws: float
    FFMC: float
    DMC: float
    ISI: float
    whatsapp_number: str


# ===========================
# FUNÇÕES DE UTILIDADE
# ===========================

def send_whatsapp_message(to: str, message: str):
    """Envia uma mensagem de texto via WhatsApp API."""
    url = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message},
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        print(f"✅ Mensagem enviada para {to}")
    else:
        print(f"❌ Erro ao enviar mensagem: {response.text}")


def upload_media(image_path: str) -> str:
    """Faz upload da imagem e retorna o media_id."""
    mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    file_name = os.path.basename(image_path)

    url = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    with open(image_path, "rb") as f:
        files = {"file": (file_name, f, mime_type)}
        response = requests.post(url, headers=headers, files=files, data=data)

    response.raise_for_status()
    return response.json()["id"]


def send_whatsapp_image(to: str, image_path: str, caption: str = "Alerta de incêndio detectado!"):
    """Envia uma imagem via WhatsApp API."""
    media_id = upload_media(image_path)
    url = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {"id": media_id, "caption": caption},
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print(f"📸 Imagem enviada para {to}")


# ===========================
# ENDPOINTS
# ===========================

@app.get("/health")
def health():
    """Verifica se o servidor está ativo."""
    return {"status": "healthy"}


@app.post("/predict/")
async def predict_fwi(request: PredictRequest):
    """Prevê o Fire Weather Index (FWI) e envia via WhatsApp."""
    data = np.array([[request.Temperature, request.Ws, request.FFMC, request.DMC, request.ISI]])
    scaled = scaler.transform(data)
    prediction = float(model_reg.predict(scaled)[0])

    message = f"🔥 Previsão de FWI: {prediction:.2f}"
    send_whatsapp_message(request.whatsapp_number, message)

    return JSONResponse(content={"FWI_estimation": round(prediction, 2)})


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    """Detecta incêndios em uma imagem usando YOLO e envia alerta via WhatsApp."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = model_yolo(img)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        detections.append({
            "class": model_yolo.names[cls_id],
            "confidence": round(conf, 4),
            "box": [x1, y1, x2, y2]
        })

    output_path = "detected_fire.jpg"
    cv2.imwrite(output_path, img)

    if detections:
        send_whatsapp_image("xxx", output_path)

    return JSONResponse(content={"detections": detections})


# ===========================
# EXECUÇÃO LOCAL / NGROK
# ===========================

if __name__ == "__main__":
    nest_asyncio.apply()
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(NGROK_PORT).public_url
    print(f"🚀 Servidor rodando em: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=NGROK_PORT)
