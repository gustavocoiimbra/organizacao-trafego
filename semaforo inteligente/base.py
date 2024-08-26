import time

class Camera:
    def __init__(self, posicao):
        self.posicao = posicao
    
    def verifica_trafego(self):
        # Simulando a captura de dados da camera
        # Retorna informações sobre os veículos, pedestres, etc
        return {"veiculos": 5, "pedestres": 2}
    

class SemaforoInteligente:
    def __init__(self, cameras):
        self.cameras = cameras
        self.duracao_sinal_verde = 30
    
    def ajustar_sinal(self):
        total_veiculos = 0
        total_pedestres = 0

        # Captura dados de todas as cameras
        for camera in self.cameras:
            dados = camera.verifica_trafego()
            total_veiculos += dados["veiculos"]
            total_pedestres += dados["pedestres"]
        
        if total_veiculos > 10:
            self.duracao_sinal_verde -= 5
        
        else:
            self.duracao_sinal_verde += 5
        
        # Limita o tempo de sinal verde entre 10 e 60 segundos
        self.duracao_sinal_verde = max(10, min(self.duracao_sinal_verde, 60))
    
    def iniciar(self):
        while True:
            self.ajustar_sinal()
            print(f"Sinal verde por {self.duracao_sinal_verde} segundos")
            time.sleep(self.duracao_sinal_verde)
            print("Sinal Vermelho") 
            time.sleep(5) # Tempo de sinal vermelho fixo

camera1 = Camera("Interseccao A")
camera2 = Camera("Interseccao B")

semaforo_inteligente = SemaforoInteligente([camera1, camera2])

semaforo_inteligente.iniciar()

