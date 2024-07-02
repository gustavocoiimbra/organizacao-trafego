import cv2 as cv
from ultralytics import YOLO
import random
import pandas as pd
from datetime import datetime
import streamlit as st
import os

MODEL_SOURCE_PATH = r'runs\detect\train\Detect-Accident-Non-Accident.pt' 

# Definindo a largura e a altura dos frames
LARGURA_FRAME = 640
ALTURA_FRAME = 480

LIMIAR_CONFIANCA = 0.7 # Limiar de confiança

# Se SKIP_FRAMES for 2, a cada 2 frames será processado
SKIP_FRAMES = 2
frame_count = 0

# Carregando o modelo pré-treinado YOLOv8n
model = YOLO(MODEL_SOURCE_PATH)

# Obtendo o nome de todas as classes do modelo
lista_classes = list(model.model.names.values())

# Obtendo o número máximo de classes detectadas pelo modelo
num_classes = len(model.model.names)

# Vamos gerar cores aleatórias para as classes
cores_deteccao = []
for i in range(num_classes):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    cores_deteccao.append((b, g, r))
    
# Inicializar DataFrame para armazenar os resultados
df = pd.DataFrame(columns=['local', 'horario', 'frame', 'classe', 'confiança'])

def process_video(source_path: str | int = 0) -> None:
    global frame_count, df

    # Carregando o vídeo
    cap = cv.VideoCapture(source_path)

    with st.expander("Visualização do Vídeo", expanded=False):
        while cap.isOpened():
            # Capturando frame a frame
            ret, frame = cap.read()

            if not ret:
                print("FIM!")
                break

            frame_count += 1
            if frame_count % SKIP_FRAMES != 0:
                continue

            # Redimensionando o frame
            frame = cv.resize(frame, (LARGURA_FRAME, ALTURA_FRAME))

            # Realizando a detecção de objetos no frame
            deteccoes = model.track(source=[frame], conf=LIMIAR_CONFIANCA, save=False, iou=0.70, imgsz=640, classes=0, device='cpu')

            # Convertendo a saída do modelo para um numpy array
            if len(deteccoes) != 0:
                for deteccao in deteccoes:
                    caixas = deteccao.boxes
                    for caixa in caixas:
                        id_classe = int(caixa.cls[0])
                        confianca = float(caixa.conf[0])
                        bb = caixa.xyxy[0]

                        # Desenhando uma caixa delimitadora ao redor do objeto detectado
                        cv.rectangle(frame,
                                    (int(bb[0]), int(bb[1])),
                                    (int(bb[2]), int(bb[3])),
                                    cores_deteccao[id_classe],
                                    3)
                        
                        # Exibindo o nome da classe e a confiança da detecção
                        fonte = cv.FONT_HERSHEY_COMPLEX
                        cv.putText(
                            frame,
                            lista_classes[int(id_classe)]
                            + " "
                            + str(round(confianca, 3))
                            + "",
                            (int(bb[0]), int(bb[1]) - 10),
                            fonte,
                            1,
                            (255, 255, 255),
                            2,
                        )
                        
                        # Adicionando os resultados ao DataFrame usando concat
                        new_row = pd.DataFrame([{
                            'local': 'Rua X, Setor Y, Cidade Z',
                            'horario': datetime.now(),
                            'frame': frame_count,
                            'classe': lista_classes[int(id_classe)],
                            'confiança': f"{confianca * 100:.2f}%",
                        }])
                        df = pd.concat([df, new_row], ignore_index=True)
                        
            # Atualizando o conteúdo do expander com o frame atual
            st.image(frame, channels="BGR", caption=f"Frame {frame_count}")

    cap.release() # Libera a captura de vídeo
    
# Interface Streamlit
st.title('Detecção de Acidentes em Vídeo')

# Carregar o vídeo
video_file = st.file_uploader("Selecione um vídeo para processar") # WEB CAM = 0

if video_file is not None:
    # Salvar o vídeo temporariamente
    with open('temp_video.mp4', 'wb') as f:
        f.write(video_file.getvalue())

    # Processar o vídeo
    process_video('temp_video.mp4')

    # Exibir DataFrame ao final da detecção
    st.subheader('Resultados da Detecção')
    st.dataframe(df, width=1500)  # Ajuste a largura conforme necessário

    # Remover o vídeo temporário
    os.remove('temp_video.mp4')