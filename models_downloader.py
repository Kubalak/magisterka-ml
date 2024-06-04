import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from multiprocessing import Queue, Process
from queue import Empty
from detection_utils.tensor.model_download_utils import download_archive, untar_archive


def worker(q: Queue, r: Queue):
    while 1:
        name, url = q.get()
        r.put(f"Pobieranie archiwum modelu {name}...")
        if download_archive(url):
            r.put("Rozpakowywanie archiwum")
            if untar_archive():
                r.put("Wypakowano pomyslnie!")
            else:
                r.put("Wypakowywanie nie powiodlo sie!")
        else:
            r.put("Nie udalo sie pobrac archiwum!")


if __name__ == "__main__":
    df = pd.read_csv('models.csv')

    names_dict = {row["name"]: row["link"] for _, row in df.iterrows()}

    if len(names_dict) != df['name'].count():
        raise RuntimeError("Zdublowane wartosci nazw!")

    queue = Queue()
    rqueue = Queue()
    process = Process(target=worker, args=(queue, rqueue))

    root = tk.Tk()
    root.title('Models downloader')
    root.geometry('350x200')

    label = ttk.Label(text="Wybierz model do pobrania")
    label.place(x=20, y=30)

    combo = ttk.Combobox(values=df["name"].to_list(), width=50)
    combo.set(df["name"].to_list()[0])
    combo.place(x=20, y=50)

    button = ttk.Button(text="Pobierz!", command=lambda: queue.put(
        (combo.get(), names_dict[combo.get()])))
    button.place(x=140, y=80)

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
