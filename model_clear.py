import tkinter as tk
from tkinter import ttk, messagebox
import os, shutil


def clear_model_dir(modelname):
    model_dir = os.path.join("workspace", "models", modelname)
    if os.path.exists(model_dir) and os.path.isdir(model_dir) and modelname != "":
        files = os.listdir(model_dir)
        for file in filter(lambda z: z != "pipeline.config", files):
            path = os.path.join(model_dir, file)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
        messagebox.showinfo("Informacja", "Usunięto z powodzeniem!")
    else:
        messagebox.showerror("Błąd", "Wskazany folder nie istnieje!")
    


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model clearer")
    root.geometry("400x400")
    
    label = ttk.Label(text="Wybierz model do oczyszczenia")
    label.place(x=20,y=30)
    
    comobox = ttk.Combobox(values=[*filter(lambda z: os.path.exists(os.path.join("workspace", "models", z, "pipeline.config")), os.listdir("workspace/models"))], width=50)
    comobox.place(x=20,y=50)
    
    button = ttk.Button(text="Wyczyść!", command=lambda: clear_model_dir(comobox.get()))
    button.place(x=140,y=80)
    
    root.mainloop()
