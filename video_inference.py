import cv2
import numpy as np
import tensorflow as tf

# Função para desenhar as detecções no frame
def draw_detections(frame, boxes, scores, classes, labels, conf_threshold=0.5):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= conf_threshold:
            # Convertendo coordenadas relativas para absolutas
            xmin, ymin, xmax, ymax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Desenhando bounding box e label
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
    
    return frame

# Carregar o modelo TFLite
interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

# Obter detalhes das entradas e saídas do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir labels (substitua pela lista de labels do seu modelo)
labels = ["ze"]

# Configuração do vídeo de entrada e saída
video_input_path = "VID_20240830_123707.mp4"
video_output_path = "output_video.mp4"

# Abrir o vídeo de entrada
cap = cv2.VideoCapture(video_input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Criar o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# Loop para processar o vídeo frame a frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processar a imagem de entrada (redimensionar e normalizar)
    input_size = input_details[0]['shape'][1:3]  # Geralmente é (320, 320) ou similar
    img_resized = cv2.resize(frame, (input_size[1], input_size[0]))
    img_normalized = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    # Realizar inferência
    interpreter.set_tensor(input_details[0]['index'], img_normalized)
    interpreter.invoke()

    # Extrair a saída (um tensor que contém caixas, classes e scores)
    detections = interpreter.get_tensor(output_details[0]['index'])[0]  # (300, 6)

    # Dividir o tensor de detecções
    boxes = detections[:, 0:4]  # Coordenadas das caixas
    scores = detections[:, 4]    # Confidências (scores)
    classes = detections[:, 5]   # Classes

    # Desenhar detecções no frame
    frame_with_detections = draw_detections(frame, boxes, scores, classes, labels)

    # Escrever o frame no vídeo de saída
    out.write(frame_with_detections)

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()