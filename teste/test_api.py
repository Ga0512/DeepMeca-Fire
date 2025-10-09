import cv2
import time
import requests
from tqdm import tqdm

# =============================
# CONFIGURAÇÕES
# =============================
BASE_URL = "http://127.0.0.1:9192"  # troque pelo teu ngrok se quiser testar remotamente
VIDEO_PATH = "data/wildfire.mp4"
OUTPUT_PATH = "result/wildfire.mp4"

# =============================
# FUNÇÃO PRINCIPAL
# =============================
def process_video(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎥 Processando vídeo ({total_frames} frames)...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

        try:
            response = requests.post(f"{BASE_URL}/detect/", files=files, timeout=30)
            response.raise_for_status()
            detections = response.json().get("detections", [])

            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["class"]
                conf = det["confidence"]
                color = (0, 255, 0) if label.lower() == "fire" else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        except Exception as e:
            print("⚠️ Erro ao enviar frame:", e)

        out.write(frame)
        time.sleep(0.05)  # para evitar sobrecarregar o servidor

    cap.release()
    out.release()
    print(f"✅ Vídeo processado salvo em: {output_path}")

# =============================
# EXECUÇÃO
# =============================
if __name__ == "__main__":
    process_video(VIDEO_PATH, OUTPUT_PATH)
