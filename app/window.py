from tkinter import *

class TextWindow:

    def __init__(self, master):
        self.root = master

        frame = Frame(master)
        frame.pack()
        master.title("Monitor de gestos")

        self.label = Label(master, font=("Times New Roman", 80))
        self.label.configure(wraplength=600)
        self.label.pack()

        self.update_clock()

    def update_clock(self):
        input_file = open("output.txt")
        self.label.configure(text=input_file.read())
        input_file.close()
        self.root.after(100, self.update_clock)

root = Tk()

app = TextWindow(root)

root.mainloop()
root.destroy()