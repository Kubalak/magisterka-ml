import os
import shutil
import tarfile
import requests
import pandas as pd
import tkinter as tk
from tkinter import ttk
from multiprocessing import Queue, Process
from queue import Empty

def download_archive(url):
    response = requests.get(url, stream=True)
    if response.ok:
        with open("model_tmp_local", "wb") as archive:
            shutil.copyfileobj(response.raw, archive)
        return True
    return False


def unzip_archive(path='model_tmp_local'):
    if os.path.exists(path):
        file = tarfile.open(path)
        print("Files:")
        files = file.getnames()
        print(*files, sep='\n')
        file.extractall(os.path.join('workspace', 'pre_trained_models'))
        model_dir = files[0].split('/')[0]
        print(model_dir)
        if not os.path.exists(os.path.join('workspace', 'models', model_dir)):
            os.makedirs(os.path.join('workspace', 'models', model_dir), exist_ok=True)
            shutil.copy2(os.path.join('workspace', 'pre_trained_models', model_dir, 'pipeline.config'), os.path.join('workspace', 'models', model_dir, 'pipeline.config'))
        return True
    else:
        return False


def worker(q:Queue, r:Queue):
    while 1:
        name, url = q.get()
        r.put(f"Pobieranie archiwum modelu {name}...")
        if download_archive(url):
            r.put("Rozpakowywanie archiwum")
            if unzip_archive():
                r.put("Wypakowano pomyslnie!")
            else:
                r.put("Wypakowywanie nie powiodlo sie!")
        else:
            r.put("Nie udalo sie pobrac archiwum!")


if __name__ == "__main__":
    df = pd.read_csv('models.csv')
    
    names_dict = {row["name"]:row["link"] for _,row in df.iterrows()}

    if len(names_dict) != df['name'].count():
        raise RuntimeError("Zdublowane wartosci nazw!")
    
    queue = Queue()
    rqueue = Queue()
    process = Process(target=worker, args=(queue,rqueue))
    
    root = tk.Tk()
    root.title('Models downloader')
    root.geometry('350x200')
    
    label = ttk.Label(text="Wybierz model do pobrania")
    label.place(x=20,y=30)
    
    combo = ttk.Combobox(values=df["name"].to_list(), width=50)
    combo.set(df["name"].to_list()[0])
    combo.place(x=20,y=50)
    
    button = ttk.Button(text="Pobierz!", command=lambda: queue.put((combo.get(), names_dict[combo.get()])))
    button.place(x=140,y=80)
    
    statusvar = tk.StringVar()
    statusvar.set("Nieaktywny")
    
    sbar = tk.Label(root, textvariable=statusvar, relief=tk.SUNKEN, anchor="w")
    sbar.pack(side=tk.BOTTOM, fill=tk.X)

    def update():
        try:
            val = rqueue.get_nowait()
            statusvar.set(val)
        except Empty:
            pass
        root.after(100, update)
        
    process.start()
    root.after(100, update)
    root.mainloop()
    
    process.kill()
    process.join()