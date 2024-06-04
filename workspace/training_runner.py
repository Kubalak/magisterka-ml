import re
import psutil
import cpuinfo
import os, sys
import platform
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from tkinter import messagebox
from multiprocessing import Process
from eval_runner import run_evaluation
# from object_detection.utils.config_util import get_configs_from_pipeline_file
from detection_utils.tensor.model_download_utils import download_archive, untar_archive


SCRIPT_NAME = "model_main_tf2.py"


def human_readable(size:int):
    index = 0
    sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    while size//1024 > 1:
        size /= 1024
        index += 1
    return (size, sizes[index])


def get_system_info() -> str:
    """### Returns formatted system information.
    System info includes the following:
    - OS
    - Architecture
    - CPU name
    - Core count
    - CPU frequency
    - CPU usage (at the moment of measurement)
    - RAM total
    - RAM used
    - RAM available
    - GPU's list

    Returns:
        str: Formatted system information.
    """
    uname = platform.uname()
    arch = platform.architecture()
    machine = platform.machine()
    info_cpu = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()
    ram_total = human_readable(mem.total)
    ram_avail = human_readable(mem.available)
    ram_used = human_readable(mem.used)
    cpu_usage = psutil.cpu_percent()
    cpu_freq = psutil.cpu_freq()
    regex = re.compile(r'^GPU \d+:.+\(')
    
    gpu_info = subprocess.getoutput("nvidia-smi -L").split('\n')
    gpus = []
    for info in gpu_info:
        matched = regex.match(info)
        if matched is not None:
            gpus.append(matched.group(0)[:-1].split(':'))
    
    info = f'+{"System info".center(62, "-")}+\n'
    info += "| Uname".ljust(17)+f"{uname.system} {uname.release}".ljust(46) + '|\n'
    info += "| Architecture".ljust(17) + arch[0].ljust(46) + '|\n'
    info += "| Machine".ljust(17) + machine.ljust(46) + '|\n'
    info += "| CPU".ljust(17)+ info_cpu['brand_raw'].ljust(46)+  '|\n'
    info += "| Core count".ljust(17)+  f"{info_cpu['count']}".ljust(46)+  '|\n'
    info += "| CPU frequency".ljust(17) +  f"Max {cpu_freq.max:.2f} MHz Current {cpu_freq.current:.2f} MHz".ljust(46)+  '|\n'
    info += "| CPU usage".ljust(17)+  f"{cpu_usage:.2f} %".ljust(46)+  '|\n'
    info += "| RAM total".ljust(17)+  f"{ram_total[0]:.2f} {ram_total[1]}".ljust(46)+  '|\n'
    info += "| RAM used".ljust(17)+  f"{ram_used[0]:.2f} {ram_used[1]}".ljust(46)+  '|\n'
    info += "| RAM available".ljust(17)+  f"{ram_avail[0]:.2f} {ram_avail[1]}".ljust(46)+  '|\n'
    
    for gpu_id,  gpu_name in gpus:
        info += f'| {gpu_id}'.ljust(17) + f"{gpu_name.strip()}".ljust(46) + '|\n'
        
    info += f'+{"End system info".center(62, "-")}+'
    return info

    
def train_model(modelname:str, turn_off:bool, evaluate:bool):
    """Executes and controls training process. 
    Kills training if `nan` is found in total loss.

    NOTE: This solution is designed to work in both Windows and Linux environment (although Linux not tested yet).

    Arguments:
    modelname (bool): Name of the model (directory).
    turn_off (bool): If set to `True` computer will turn off after training (and evaluation if set).
    evaluae (bool): If set to `True` evaluaion script will be run after training see `eval_runner.py`.
    """
    # model_config = get_configs_from_pipeline_file(os.path.join("models", modelname, "pipeline.config"))
    # model_config['train_config'].fine_tune_checkpoint
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
        logfile.write(get_system_info().encode('utf-8'))
        logfile.write(f'\npython {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --model_dir="models/{modelname}" --checkpoint_every_n=1000 --num_workers=1 --alsologtostderr\n'.encode('utf-8'))
        
    while subproc.poll() is None:
        c = subproc.stderr.readline()
        sys.stdout.write(c.decode('utf-8',  errors='ignore'))
        with open(f"logs/{modelname}_{time}-stderr.log", "ab") as logfile:
            logfile.write(c)
            line = c.decode('utf-8', errors='ignore')
            if line.find("'Loss/total_loss': nan") != -1:
                print("Killing broken learning process")
                logfile.write("Killing broken learning process...\n".encode('utf-8'))
                if os.name == 'nt':
                    subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
                else:
                    subprocess.Popen(f"kill -TERM {subproc.pid}")
                training_killed = True
            elif line.find("RESOURCE_EXHAUSTED: Out of memory") != -1 or line.find("UnicodeDecodeError:") != -1 or line.find("RESOURCE_EXHAUSTED: failed to allocate memory") != -1:
                training_failed = True
                logfile.write("Killing learning process that produced error...\n".encode('utf-8'))
                if os.name == 'nt':
                    subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
                else:
                    subprocess.Popen(f"kill -KILL {subproc.pid}")
            
    errors.close()
    
    if training_failed:
        messagebox.showerror("Podczas uczenia wystąpił błąd!", f"Sprawdź plik 'logs/{modelname}_{time}-stderr.log' aby uzyskać więcej informacji.")
        return
    
    if not training_killed and subproc.returncode == 0 and evaluate:
        run_evaluation(modelname, time)
    
    if not training_killed and turn_off:
        root = tk.Tk()
        root.title("Info")
        root.geometry("200x100")
        root.resizable(width=False, height=False)
        timer = tk.IntVar(value=10)
        txt = ttk.Label(text=f"Komputer wylaczy sie za {timer.get()}s")
        btn = ttk.Button(text="Anuluj", command=root.destroy)
        
        def countdown():
            if timer.get() > 0:
                timer.set(timer.get() - 1)
                txt.config(text=f"Komputer wylaczy sie za {timer.get()}s")
                root.after(1000, countdown)
            else:
                root.destroy()
        
        txt.place(x=20,y=20)
        btn.place(x=60,y=60)
        root.after(1000, countdown)
        
        root.mainloop()
        if timer.get() == 0:
            if os.name == 'nt':
                os.system("shutdown /s /t 1")
            else:
                os.system("init 0")

        
def is_model_empty(modelname):
    """Tells whether model dir passed via `modelname` is empty (contains only `pipeline.config` file)."""
    return (not os.path.exists(os.path.join("models", modelname, "checkpoint"))) and os.path.exists(os.path.join("models", modelname, "pipeline.config"))


if __name__ == "__main__":
    root = tk.Tk()
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
    
    def start_training(modelname, turn_off, evaluate):
        root.destroy()
        train_model(modelname, turn_off, evaluate)

    button = ttk.Button(text="Start", command=lambda: start_training(comobox.get(), shutdown.get(), evaluate.get()))
    button.place(x=140,y=140)
    
    root.mainloop()
    
    
    