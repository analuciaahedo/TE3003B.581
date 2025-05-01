import cv2
import numpy as np
import os
from ultralytics import YOLO
from sort import Sort

# Nombre del video de entrada
video_path = 'Fuchibol.mp4'

# Verificar si el archivo existe
if not os.path.exists(video_path):
    print(f"Error: El archivo '{video_path}' no existe.")
    exit()

# Cargar el modelo YOLO
model = YOLO("yolov8s.pt")

# Inicializar SORT
tracker = Sort()

# Cargar video de entrada
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Escritor de video
out = cv2.VideoWriter('Fuchibol_analizado_n.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Procesar cada frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=960)

    detections = []
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if int(cls) == 32:  # Clase 32 = sports ball en COCO
            detections.append([x1, y1, x2, y2, conf])

    # Convertir a NumPy array para SORT
    if len(detections) > 0:
        dets_np = np.array(detections)
    else:
        dets_np = np.empty((0, 5))

    # Aplicar seguimiento
    tracks = tracker.update(dets_np)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()

# Reproducir automáticamente el video final
print("Procesamiento terminado. Reproduciendo el video analizado")

cap_out = cv2.VideoCapture('video_salida.mp4')
while cap_out.isOpened():
    ret, frame = cap_out.read()
    if not ret:
        break
    cv2.imshow('Video Final', frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap_out.release()
cv2.destroyAllWindows()
