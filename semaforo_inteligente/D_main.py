# Desativando os avisos no notebook para manter células de saída limpas
import warnings
warnings.filterwarnings('ignore')

# Importação das bibliotecas necessárias
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from PIL import Image
from ultralytics import YOLO
import threading

# Definir dimensões constantes para o vídeo (largura x altura)
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Variáveis de controle do semáforo
semaforoAberto_esquerda = True  # Inicia com o semáforo verde
semaforoAberto_direita = True   # Inicia com o semáforo verde

# Tempos mínimos e máximos para o semáforo
tempo_verde_min = 30  # Tempo mínimo para o semáforo verde (segundos)
tempo_verde_max = 50  # Tempo máximo para o semáforo verde (segundos)
tempo_vermelho = 30   # Tempo fixo para o semáforo vermelho (segundos)

# Variáveis para armazenar a contagem de veículos detectados
fluxo_veiculos_esquerda = 0
fluxo_veiculos_direita = 0

# Função para alternar os semáforos de acordo com o fluxo
def alternar_semaforo(area):
    global semaforoAberto_esquerda, semaforoAberto_direita, fluxo_veiculos_esquerda, fluxo_veiculos_direita

    if area == "esquerda":
        semaforoAberto_esquerda = not semaforoAberto_esquerda
        print(f'SEMÁFORO ESQUERDA {"VERDE" if semaforoAberto_esquerda else "VERMELHO"}')

        if semaforoAberto_esquerda:
            # Ajusta o tempo verde de acordo com o fluxo de veículos
            tempo_verde_esquerda = max(tempo_verde_min, min(tempo_verde_max, fluxo_veiculos_esquerda * 2))
            print(f"Tempo verde ESQUERDA ajustado para: {tempo_verde_esquerda}s")
            threading.Timer(tempo_verde_esquerda, alternar_semaforo, args=("esquerda",)).start()
        else:
            # Tempo fixo de vermelho
            threading.Timer(tempo_vermelho, alternar_semaforo, args=("esquerda",)).start()

    elif area == "direita":
        semaforoAberto_direita = not semaforoAberto_direita
        print(f'SEMÁFORO DIREITA {"VERDE" if semaforoAberto_direita else "VERMELHO"}')

        if semaforoAberto_direita:
            # Ajusta o tempo verde de acordo com o fluxo de veículos
            tempo_verde_direita = max(tempo_verde_min, min(tempo_verde_max, fluxo_veiculos_direita * 2))
            print(f"Tempo verde DIREITA ajustado para: {tempo_verde_direita}s")
            threading.Timer(tempo_verde_direita, alternar_semaforo, args=("direita",)).start()
        else:
            # Tempo fixo de vermelho
            threading.Timer(tempo_vermelho, alternar_semaforo, args=("direita",)).start()

# Inicializar os semáforos
threading.Timer(tempo_verde_min, alternar_semaforo, args=("esquerda",)).start()
threading.Timer(tempo_verde_min, alternar_semaforo, args=("direita",)).start()

# Função para atualizar o estado do semáforo com base na quantidade de veículos
def updateSemaforo(number_vehiche: int, heavy_traffic_threshold: int):
    if number_vehiche > heavy_traffic_threshold:
        traffic_intensity = "Pesado"
    else:
        traffic_intensity = "Leve"
    return traffic_intensity

# ARQUIVOS
input_file = 'arquivos/cars3.mp4'
path_model = 'runs/detect/detect_vehicle/weights/best.pt'
best_model = YOLO(path_model)

# CONFIG PRECISÃO
LIMIAR_CONFIANCA = 0.3
threshold_NMS = 0.4
lane_threshold = 300 #609

# CONFIG DO PROCESSAMENTO DO VIDEO
frame_count = 0
SKIP_FRAMES = 2

# CARREGAR VIDEO
cap = cv.VideoCapture(input_file)

# Defina o limite para considerar o tráfego como pesado
heavy_traffic_threshold = 10

# Cars 2
vertice1 = np.array([[556, 678], [793, 772], [23, 1038], [23, 784]], dtype=np.int32)
vertice2 = np.array([[1162, 663], [1126, 869], [1902, 929], [1905, 681]], dtype=np.int32)

# Cars3
novos_vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
novos_vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Cars4
vertice4 = np.array([[741, 43], [962, 43], [1756, 1046], [156, 1037]], dtype=np.int32)

# Cars5
vertice5 = np.array([(520, 367), (692, 367), (744, 923), (8, 927)], dtype=np.int32)
vertice6 = np.array([(784, 375), (880, 355), (1596, 747), (1180, 927)], dtype=np.int32)

# Definir as áreas de detecção (polígonos)
# Faixa 1
vertice5 = np.array([[947, 396], [1065, 360], [1908, 841], [1438, 1075]], dtype=np.int32)
# Faixa 2
vertice6 = np.array([[656, 378], [847, 372], [920, 1063], [11, 1060]], dtype=np.int32)

# Defina as posições das anotações de texto na imagem
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Defina fonte, escala e cores para as anotaçõess
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

# Continuando com o loop de processamento de vídeo
while cap.isOpened():
    # Capturando frame a frame
    ret, frame = cap.read()

    if not ret and type(frame) == type(None):
        print("VIDEO NULL")
        break
    
    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    # Redimensionar o frame para a dimensão desejada
    #frame = cv.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Inicializar as variáveis de intensidade de tráfego
    traffic_intensity_left = "Leve"
    traffic_intensity_right = "Leve"

    # Realizando a detecção de objetos no frame
    results = best_model.track(source=[frame],
                               conf=LIMIAR_CONFIANCA, 
                               save=False, iou=0.50, 
                               imgsz=640)

    # Cópia do frame original para sobrepor as detecções
    processed_frame = frame.copy()

    # Verificar se existem detecções
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        bounding_boxes = results[0].boxes.xyxy
        ids_detectados = results[0].boxes.id  # IDs dos objetos detectados

        # Garantir que há caixas e IDs para processar
        if bounding_boxes is not None and ids_detectados is not None:
            # Inicialize contadores para veículos em cada faixa
            vehicles_in_left_lane = 0
            vehicles_in_right_lane = 0

            # Percorrer as caixas delimitadoras para realizar contagem de veículos
            for box, obj_id in zip(bounding_boxes, ids_detectados):
                # Verificar se o veículo está na faixa da esquerda
                if box[0] < lane_threshold and not semaforoAberto_esquerda:
                    print("Ok")
                    vehicles_in_left_lane += 1
                    traffic_intensity_left = updateSemaforo(vehicles_in_left_lane, heavy_traffic_threshold)
                    # Desenhe o quadrilátero e exiba as detecções apenas se o semáforo estiver vermelho
                    cv.polylines(processed_frame, [novos_vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv.rectangle(processed_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                # Verificar se o veículo está na faixa da direita
                elif box[0] >= lane_threshold and not semaforoAberto_direita:
                    vehicles_in_right_lane += 1
                    traffic_intensity_right = updateSemaforo(vehicles_in_right_lane, heavy_traffic_threshold)
                    # Desenhe o quadrilátero e exiba as detecções apenas se o semáforo estiver vermelho
                    cv.polylines(processed_frame, [novos_vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv.rectangle(processed_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Atualizar o fluxo de veículos detectado
        fluxo_veiculos_esquerda = vehicles_in_left_lane
        fluxo_veiculos_direita = vehicles_in_right_lane
    
    # Exibir semáforo para a faixa esquerda
    if semaforoAberto_esquerda:
        semaforoImage_left = cv.imread("arquivos/sTrue.png")  # Semáforo verde
    else:
        # Semáforo vermelho, exibir frame com detecções para a esquerda
        semaforoImage_left = cv.imread("arquivos/sFalse.png")  # Semáforo vermelho
    
    # Exibir semáforo para a faixa direita
    if semaforoAberto_direita:
        semaforoImage_right = cv.imread("arquivos/sTrue.png")  # Semáforo verde
    else:
        # Semáforo vermelho, exibir frame com detecções para a direita
        semaforoImage_right = cv.imread("arquivos/sFalse.png")  # Semáforo vermelho

    # Mostrar os semáforos
    cv.imshow("Semáforo Esquerda", semaforoImage_left)
    cv.imshow("Semáforo Direita", semaforoImage_right)
    
    # Exibir a contagem de veículos para cada faixa
    cv.putText(processed_frame, f'Veiculos acumulados na esquerda: {vehicles_in_left_lane}', 
               (10, 150), font, font_scale, font_color, 2, cv.LINE_AA)
    cv.putText(processed_frame, f'Veiculos acumulados na direita: {vehicles_in_right_lane}', 
               (700, 150), font, font_scale, font_color, 2, cv.LINE_AA)

    # Exibir o frame processado apenas quando um dos semáforos estiver vermelho
    if not semaforoAberto_esquerda or not semaforoAberto_direita:
        cv.imshow('Real-time Analysis', processed_frame)
    else:
        cv.imshow('Real-time Analysis', frame)

    # Parar o loop ao pressionar 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas
cap.release()
cv.destroyAllWindows()