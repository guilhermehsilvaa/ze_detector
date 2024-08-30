import cv2
import os

img_number = len(os.listdir('C:\\Users\\gui_h\\OneDrive\\Documentos\\ze_detector\\datasets\\images\\'))

def video_to_frames(video_path, output_folder, frame_interval=10):
    # Verifica se a pasta de saída existe, caso contrário, cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    
    # Verifica se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    frame_count = 0
    saved_frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            # Salva o frame se ele for um múltiplo de frame_interval
            frame_name = os.path.join(output_folder, f"frame_{(saved_frame_count+img_number):03d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    print(f"Extração concluída. {saved_frame_count} frames foram salvos na pasta {output_folder}.")

# Exemplo de uso
video_path = 'C:\\Users\\gui_h\\Downloads\\VID_20240830_123707.mp4'
output_folder = "C:\\Users\\gui_h\\OneDrive\\Documentos\\ze_detector\\datasets\\images\\"
video_to_frames(video_path, output_folder)