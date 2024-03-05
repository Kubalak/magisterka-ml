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

def start_training(modelname):
    global process
    process = Process(target=train_model, args=(modelname,))
    process.start()
    root.destroy()
    
def train_model(modelname):
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
        if c.decode('utf-8', errors='ignore').find("'Loss/classification_loss': nan") != -1:
            print("Killing broken learning process")
            logfile.write("Killing broken learning process...\n".encode('utf-8'))
            subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
            training_killed = True
        with open(f"logs/{modelname}_{time}-stderr.log", "ab") as logfile:
            logfile.write(c)

    errors.close()
    
    if not training_killed:
        run_evaluation(modelname, time)
    


if __name__ == "__main__":
    root.title("Model training runner")
    root.geometry("400x400")
    
    label = ttk.Label(text="Wybierz model do nauczenia")
    label.place(x=20,y=30)
    
    comobox = ttk.Combobox(values=[*filter(lambda z: os.path.exists(os.path.join("models", z, "pipeline.config")), os.listdir("models"))], width=50)
    comobox.place(x=20,y=50)
    
    button = ttk.Button(text="Start", command=lambda: start_training(comobox.get()))
    button.place(x=140,y=80)
    
    root.mainloop()
    
    if process is not None:
        process.join()
    
    