import tkinter as tk
from tkinter import ttk
from multiprocessing import Process
import subprocess
import os, sys
from datetime import datetime
from eval_runner import run_evaluation

SCRIPT_NAME = "model_main_tf2.py"

process = None
root = tk.Tk()

def start_training(modelname, turn_off=False):
    global process
    process = Process(target=train_model, args=(modelname,turn_off))
    process.start()
    root.destroy()
    
def train_model(modelname, turn_off):
    time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    errors = open(f"logs/{modelname}_{time}-stdout.log", "wb")

    training_killed = False
    
    subproc = subprocess.Popen(
        f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --model_dir="models/{modelname}" --checkpoint_every_n=1000 --num_workers=1 --alsologtostderr',
        shell=True,
        stdout=errors,
        stderr=subprocess.PIPE
    )
    with open(f"logs/{modelname}_{time}-stderr.log", "wb") as logfile:
        logfile.write(f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --model_dir="models/{modelname}" --checkpoint_every_n=1000 --num_workers=1 --alsologtostderr\n'.encode('utf-8'))
        
    while subproc.poll() is None:
        c = subproc.stderr.readline()
        sys.stdout.write(c.decode('utf-8',  errors='ignore'))
        with open(f"logs/{modelname}_{time}-stderr.log", "ab") as logfile:
            logfile.write(c)
            if c.decode('utf-8', errors='ignore').find("'Loss/classification_loss': nan") != -1:
                print("Killing broken learning process")
                logfile.write("Killing broken learning process...\n".encode('utf-8'))
                subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
                training_killed = True
            
    errors.close()
    
    if not training_killed:
        run_evaluation(modelname, time)
        if turn_off:
            os.system("shutdown /s /t 1")
        
def is_model_empty(modelname):
    return (not os.path.exists(os.path.join("models", modelname, "checkpoint"))) and os.path.exists(os.path.join("models", modelname, "pipeline.config"))


if __name__ == "__main__":
    root.title("Model training runner")
    root.geometry("400x400")
    
    label = ttk.Label(text="Wybierz model do nauczenia")
    label.place(x=20,y=30)
    
    comobox = ttk.Combobox(values=[*filter(is_model_empty, os.listdir("models"))], width=50)
    comobox.place(x=20,y=50)
    
    shutdown = tk.IntVar()
    
    checkbox = ttk.Checkbutton(text="Wyłącz komputer po zakończeniu", variable=shutdown, onvalue=1, offvalue=0)
    checkbox.place(x=20, y=80)
    
    button = ttk.Button(text="Start", command=lambda: start_training(comobox.get(), shutdown.get()))
    button.place(x=140,y=120)
    
    root.mainloop()
    
    if process is not None:
        process.join()
    
    