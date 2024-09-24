import cv2 as cv
from ultralytics import YOLO
import random
import pandas as pd
from datetime import datetime
import streamlit as st
import os
import imageio
import pydeck as pdk
import numpy as np
import tempfile

MODEL_SOURCE_PATH = r'runs\detect\train\Detect-Accident-Non-Accident.pt'

# Definindo a largura e a altura dos frames
LARGURA_FRAME = 640
ALTURA_FRAME = 480

# Se SKIP_FRAMES for 2, a cada 2 frames será processado
SKIP_FRAMES = 2
frame_count = 0

# Carregando o modelo pré-treinado YOLOv8n
model = YOLO(MODEL_SOURCE_PATH)

# Obtendo o nome de todas as classes do modelo
lista_classes = list(model.model.names.values())

# Obtendo o número máximo de classes detectadas pelo modelo
num_classes = len(model.model.names)

# Gerando cores aleatórias para as classes
cores_deteccao = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

# Inicializar DataFrame para armazenar os resultados e verificar se existe GIF na sessão
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['local', 'horario', 'frame', 'classe', 'confiança', 'bbox', 'longitude', 'latitude'])

if 'gif_path' not in st.session_state:
    st.session_state.gif_path = None

if 'frames' not in st.session_state:
    st.session_state.frames = []

def filtrar_bbox(df, threshold):
    filtered_rows = []
    for index, row in df.iterrows():
        bb_current = row['bbox']
        too_close = False
        for f_row in filtered_rows:
            bb_existing = f_row['bbox']
            dist = np.sqrt((bb_current[0] - bb_existing[0])**2 + (bb_current[1] - bb_existing[1])**2)
            if dist < threshold:
                too_close = True
                break
        if not too_close:
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

def process_video(source_path: str | int = 0, limiar_confianca: float = 0.7) -> None:
    global frame_count

    # Lista para armazenar os frames
    frames = []

    # Carregando o vídeo
    cap = cv.VideoCapture(source_path)

    # Obter o número total de frames do vídeo para a barra de progresso
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)  # Inicializar a barra de progresso

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
        deteccoes = model.track(source=[frame], conf=limiar_confianca, save=False, iou=0.70, imgsz=640, classes=0, device='cpu')

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

                    # Adicionando os resultados ao DataFrame original
                    new_row = pd.DataFrame([{
                        'local': 'Instituto de Informática, UFG Samambaia, Goiânia - GO',
                        'horario': datetime.now(),
                        'frame': frame_count,
                        'classe': lista_classes[int(id_classe)],
                        'confiança': f"{confianca * 100:.2f}%",
                        'bbox': [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                        'longitude': -49.266657,
                        'latitude': -16.603084
                    }])
                    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)

        # Armazenando o frame processado
        frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # Atualizar a barra de progresso
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()  # Libera a captura de vídeo

    # Certificar que a barra de progresso está completa
    progress_bar.progress(1.0)

    # Criar GIF a partir dos frames
    gif_path = "output.gif"
    imageio.mimsave(gif_path, frames, duration=0.1)
    
    st.session_state.gif_path = gif_path
    st.session_state.frames = frames
    
    return frames

# Interface Streamlit
st.title('Detecção de Acidentes em Vídeo v3.0 (Beta)')

# Adicionar um slider para o limiar de confiança
limiar_confianca = st.slider(
    'Limiar de Confiança',
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01
)

# Carregar o vídeo
video_file = st.file_uploader("Selecione um vídeo para processar")  # WEB CAM = 0

# Verifica se um novo vídeo foi carregado
if video_file is not None:
    # Resetar o estado ao carregar um novo vídeo
    if st.session_state.get('last_uploaded_file') != video_file.name or 'last_limiar' not in st.session_state or st.session_state['last_limiar'] != limiar_confianca:
        st.session_state.df = pd.DataFrame(columns=['local', 'horario', 'frame', 'classe', 'confiança', 'bbox', 'longitude', 'latitude'])
        st.session_state.gif_path = None
        st.session_state.frames = []
        st.session_state.last_uploaded_file = video_file.name
        st.session_state.last_limiar = limiar_confianca
        frame_count = 0  # Resetar o contador de frames

    # Salvar o vídeo temporariamente
    with open('temp_video.mp4', 'wb') as f:
        f.write(video_file.getvalue())

    # Processar o vídeo e gerar GIF, se ainda não tiver sido processado
    if st.session_state.gif_path is None:
        process_video('temp_video.mp4', limiar_confianca)

    # Exibir o GIF
    if st.session_state.gif_path:
        st.subheader('Visualização do Resultado em Vídeo')
        st.image(st.session_state.gif_path, caption="Vídeo gerado a partir dos frames processados")

    # Adicionar o expander com os frames detectados
    if st.session_state.frames:
        with st.expander("Visualização do Resultado Frame a Frame"):
            for i, frame in enumerate(st.session_state.frames):
                st.image(frame, caption=f"Frame {i*SKIP_FRAMES+SKIP_FRAMES}")

    # Exibir DataFrame ao final da detecção
    st.subheader('Resultados Brutos da Detecção')
    st.dataframe(st.session_state.df, width=2000)

    # Adicionar botão de download para o DataFrame original
    csv_bruto = st.session_state.df.to_csv(index=False)
    st.download_button(label="Baixar Detecções Brutas", data=csv_bruto, file_name="detecções_brutas.csv", mime="text/csv")
    
    # Slider para ajustar o threshold
    threshold_distancia = st.slider('Distância entre Caixas de Detecção', 0, 200, 50, 1)
    
    # Filtrar o DataFrame para remover bbox muito próximas
    df_filtrado = filtrar_bbox(st.session_state.df, threshold_distancia)
    
    st.subheader('Acidentes Únicos')
    st.dataframe(df_filtrado, width=2000)
    # Converter as colunas para tipo numérico (float)
    df_filtrado['latitude'] = pd.to_numeric(df_filtrado['latitude'], errors='coerce')
    df_filtrado['longitude'] = pd.to_numeric(df_filtrado['longitude'], errors='coerce')

    csv_filtrado = df_filtrado.to_csv(index=False)
    st.download_button(label="Baixar Acidentes Únicos", data=csv_filtrado, file_name="detecções_filtradas.csv", mime="text/csv")
    
    # Remover o vídeo temporário
    os.remove('temp_video.mp4')
    
    st.subheader('Mapa de Calor dos Locais de Acidentes')
    
    # Mapa de calor usando as coordenadas filtradas
    if not df_filtrado.empty:
        # Criar o gráfico do mapa de calor
        deck = pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=-16.603084,
                longitude=-49.266657,
                zoom=17,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=df_filtrado,
                    get_position="[longitude, latitude]",
                    radius=20,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    opacity=0.9,
                ),
            ],
        )

        # Exibir o mapa no Streamlit
        st.pydeck_chart(deck)
