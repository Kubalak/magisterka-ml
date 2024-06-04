import os
import sys
import subprocess


SCRIPT_NAME = "model_main_tf2.py"


def run_evaluation(modelname, time):
    """Runs evaluation in subprocess and logs output to `workspace/logs/{modelname}_{time}-(stdout/stderr).log`

    Args:
        modelname (str): Name of the model folder in `workspace/models`.
        time (str): Timestamp for log file name.
    """
    errors = open(f"logs/{modelname}_{time}-stdout.log", "ab")

    subproc = subprocess.Popen(
        f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --model_dir="models/{modelname}" --checkpoint_dir="models/{modelname}" --alsologtostderr',
        shell=True,
        stdout=errors,
        stderr=subprocess.PIPE
    )
    with open(f"logs/{modelname}_{time}-stderr.log", "ab") as logfile:
        logfile.write(
            f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --model_dir="models/{modelname}" --checkpoint_dir="models/{modelname}" --alsologtostderr\n'.encode('utf-8'))

    while subproc.poll() is None:
        c = subproc.stderr.readline()
        sys.stdout.write(c.decode('utf-8',  errors='ignore'))
        if c.decode('utf-8', errors='ignore').find("'Loss/classification_loss': nan") != -1:
            print("Killing broken learning process")
            logfile.write(
                "Killing broken learning process...\n".encode('utf-8'))
            subprocess.Popen(f"TASKKILL /F /PID {subproc.pid} /T")
        with open(f"logs/{modelname}_{time}-stderr.log", "ab") as logfile:
            logfile.write(c)

    errors.close()


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk
    from datetime import datetime

    root = tk.Tk()

    def run_eval_gui(modelname):
        time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
        root.destroy()
        run_evaluation(modelname, time)

    root.title("Model evaluation runner")
    root.geometry("400x400")

    label = ttk.Label(text="Wybierz model do ewaluacji")
    label.place(x=20, y=30)

    comobox = ttk.Combobox(values=[*filter(lambda z: os.path.exists(
        os.path.join("models", z, "checkpoint")), os.listdir("models"))], width=50)
    comobox.place(x=20, y=50)

    button = ttk.Button(
        text="Start", command=lambda: run_eval_gui(comobox.get()))
    button.place(x=140, y=80)

    root.mainloop()
