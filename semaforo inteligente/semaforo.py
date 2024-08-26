from tkinter import Tk, Canvas
from time import sleep

# Define main eindow
traffic_lights = Tk()
traffic_lights.title("Semáforo")
traffic_lights.geometry("300x400+600+150")
traffic_lights.resizable(False, False)
traffic_lights.iconbitmap("semaforo_icon.jpg")
traffic_lights.config(bg="#4ea6a1")

# Define widgets
myCanvas = Canvas(traffic_lights, height=400, width=300, bg='white')
myCanvas.pack()
myCanvas.create_rectangle(100, 50, 200, 350, fill="#4ea6a1", width=2)

# Define Functions
a = 10

while True:
    def green():
        for i in range(a):
            green = myCanvas.create_oval(100, 50, 200, 150, fill="Green", width=2)
            traffic_lights.update()
            sleep(0.50)
    
    def greenb():
        for i in range(a):
            green = myCanvas.create_oval(100, 50, 200, 150, fill="Black", width=2)
            traffic_lights.update()
            sleep(0.00001)
    
    def orange():
        for i in range(a):
            orange = myCanvas.create_oval(100, 150, 200, 250, fill="Orange", width=2)
            traffic_lights.update()
            sleep(0.05)
    
    def orangeb():
        for i in range(a):
            orange = myCanvas.create_oval(100, 150, 200, 250, fill="Black", width=2)
            traffic_lights.update()
            sleep(0.00001)
    
    def red():
        for i in range(a):
            red = myCanvas.create_oval(100, 250, 200, 350, fill="Red", width=2)
            traffic_lights.update()
            sleep(0.50)
    
    def redb():
        for i in range(a):
            red = myCanvas.create_oval(100, 250, 200, 350, fill="Black", width=2)
            traffic_lights.update()
            sleep(0.00001)
    
    green()
    greenb()
    orange()
    orangeb()
    red()
    redb()


# Looping main window
traffic_lights.mainloop()