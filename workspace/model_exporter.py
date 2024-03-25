import subprocess
import os


SCRIPT_NAME = "exporter_main_v2.py"


def export_model(modelname):
    
    os.makedirs(os.path.join("exported_models", namespace.model_dir_name), exist_ok=True)
    
    subproc = subprocess.Popen(
        f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --trained_checkpoint_dir="models/{modelname}" --output_directory="exported_models/{modelname}" --input_type=image_tensor',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    with open(f"logs/{modelname}_export.log", "wb") as logfile:
        logfile.write(f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --trained_checkpoint_dir="models/{modelname}" --output_directory="exported_models/{modelname}" --input_type=image_tensor\n'.encode('utf-8'))
        stderr, stdout = subproc.communicate()
        logfile.write(stderr)
        logfile.write(stdout)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_name", "-md", help="Model directory to export (from models dir)", type=str, required=True)
    namespace = parser.parse_args()
    
    if os.path.exists(os.path.join("models", namespace.model_dir_name, "checkpoint")):
        export_model(namespace.model_dir_name)
    else:
        print(f"No checkpoint found in {namespace.model_dir_name}!")
        exit(1)
    
