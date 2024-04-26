import os, sys
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from tkinter import messagebox
from multiprocessing import Process
from eval_runner import run_evaluation
from detection_utils.tensor.model_download_utils import download_archive, untar_archive

SCRIPT_NAME = "model_main_tf2.py"

process = None
root = tk.Tk()

def start_training(modelname, turn_off, evaluate):
    """Starts a new process to run and control training. Closes main app window.
    
    Arguments:
    modelname -- Name of the model (directory).
    turn_off -- If set to `True` computer will turn off after training (if training has not been interrupted).
    evaluate -- If set to `True` evaluaion script will be run after training see `eval_runner.py`.
    """
    global process
    process = Process(target=train_model, args=(modelname,turn_off, evaluate))
    process.start()
    root.destroy()
    
def train_model(modelname, turn_off, evaluate):
    """Executes and controls training process. 
    Kills training if `nan` is found in total loss.

    NOTE: This solution is designed to work in Windows environment.

    Arguments:
    modelname -- Name of the model (directory).
    turn_off -- If set to `True` computer will turn off after training (and evaluation if set).
    evaluae -- If set to `True` evaluaion script will be run after training see `eval_runner.py`.
    """
    if not os.path.exists(f"pre_trained_models/{modelname}/saved_model"):
        print("Pre trained model does not exist, downloading...")
        df = pd.read_csv("../models.csv")
        models = df[df['link'].str.contains(modelname)]
        models = models.reset_index()
        model = models.iloc[0]
        if download_archive(model['link']):
            if not untar_archive(pre_trained_dir="pre_trained_models", models_dir="models"):
                messagebox.showerror("Wypakowanie nie powiodło się", "Nie udało się wypakować archiwum modelu")
                return
        else:
            messagebox.showerror("Pobieranie modelu nie powiodło się", f"Nie udało się pobrać archiwum modelu {modelname}")
            return
        print("Download successfull!")
                
    
    time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    errors = open(f"logs/{modelname}_{time}-stdout.log", "wb")

    training_killed = False
    training_failed = False
    
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
            line = c.decode('utf-8', errors='ignore')
            if line.find("'Loss/total_loss': nan") != -1:
                print("Killing broken learning process")
                logfile.write("Killing broken learning process...\n".encode('utf-8'))
                subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
                training_killed = True
            elif line.find("RESOURCE_EXHAUSTED: Out of memory") != -1 or line.find("UnicodeDecodeError:") != -1 or line.find("RESOURCE_EXHAUSTED: failed to allocate memory") != -1:
                training_failed = True
            
    errors.close()
    
    if training_failed:
        messagebox.showerror("Podczas uczenia wystąpił błąd!", f"Sprawdź plik 'logs/{modelname}_{time}-stderr.log' aby uzyskać więcej informacji.")
        return
    
    if not training_killed and subproc.returncode == 0 and evaluate:
        run_evaluation(modelname, time)
    
    if not training_killed and turn_off:
        os.system("shutdown /s /t 10")
        
def is_model_empty(modelname):
    """Tells whether model dir passed via `modelname` is empty (contains only `pipeline.config` file)."""
    return (not os.path.exists(os.path.join("models", modelname, "checkpoint"))) and os.path.exists(os.path.join("models", modelname, "pipeline.config"))


if __name__ == "__main__":
    root.title("Model training runner")
    root.geometry("400x400")
    
    label = ttk.Label(text="Wybierz model do nauczenia")
    label.place(x=20,y=30)
    
    comobox = ttk.Combobox(values=[*filter(is_model_empty, os.listdir("models"))], width=50)
    comobox.place(x=20,y=50)
    
    shutdown = tk.IntVar()
    evaluate = tk.IntVar(value=1)
    
    checkbox = ttk.Checkbutton(text="Wyłącz komputer po zakończeniu", variable=shutdown, onvalue=1, offvalue=0)
    checkbox.place(x=20, y=80)

    eval_box =  ttk.Checkbutton(text="Uruchom ewaluację po zakończeniu uczenia", variable=evaluate, onvalue=1, offvalue=0)
    eval_box.place(x=20, y=100)

    button = ttk.Button(text="Start", command=lambda: start_training(comobox.get(), shutdown.get(), evaluate.get()))
    button.place(x=140,y=140)
    
    root.mainloop()
    
    if process is not None:
        process.join()
    
    