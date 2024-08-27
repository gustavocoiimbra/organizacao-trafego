from ultralytics import YOLO
import cv2

source_model = 'yolov8n.pt'
model = YOLO(source_model)
source_img = 'files/carros-image.jpg'
source_video = 'files/video1.mp4'

def printar():
    print("Estou na main!")

def process_video():
    # Carregando vídeo
    cap = cv2.VideoCapture(source_video)

    while cap.isOpened():
        # Capturando frame a frame
        ret, frame = cap.read()

        if not ret:
            print("FIM!")
            break

        cv2.imshow("Detecção de objetos", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video2():
    cap = cv2.VideoCapture(source_video)
    ret, frame = cap.read()

    while cv2.waitKey(1) < 0:
        __ret, __frame = cap.read()

        if not __ret and type(__frame) == type(None):
            print("FIM!")
            break

        

if __name__ == "__main__":
    printar()
    process_video()

