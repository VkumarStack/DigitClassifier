from tkinter import *
from PIL import Image, ImageOps
from team5_final_project import Digit_Classifier5
import torch
import numpy as np

root = Tk()
 
# Create Title
root.title(  "Paint App ")
 
# specify size
root.geometry("200x200")

xPrev, yPrev = None, None
model = Digit_Classifier5()
model.load_state_dict(torch.load('./Weights/team5_final_weights.pth'))
number = StringVar()
number.set('')

def click(event):
    global xPrev, yPrev
    xPrev, yPrev = event.x, event.y

# define function when 
# mouse double click is enabled
def paint( event ):
    global xPrev, yPrev
    # Co-ordinates.
    x, y = event.x, event.y
    if xPrev is not None and yPrev is not None:
        w.create_line((xPrev, yPrev, x, y), fill = "black", width = 5)
    xPrev, yPrev = x, y
    # specify type of display
    
def export( event ):
    if event.keysym == "Return":
        w.postscript(file = "test.eps")
        img = Image.open("test.eps")
        img = img.resize((28, 28))
        img.save("test.png")
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)
        img = np.array(img)
        img = torch.tensor(img).float().unsqueeze(0)
        number.set(torch.argmax(model(img)).item())
        # number.set(model.predict(img).item())
    elif event.keysym == "r" or event.keysym == "R":
        w.delete('all')
 
# create canvas widget.
w = Canvas(root, width = 100, height = 100)
w.configure(bg="white")

# create label
l = Label(root, textvariable=number)

# call function when double
# click is enabled.
w.bind( "<Button-1>", click )
w.bind( "<B1-Motion>", paint )
root.bind( "<KeyPress>", export )
 
w.pack()
l.pack()

mainloop()