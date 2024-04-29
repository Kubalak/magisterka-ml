import subprocess
import os,sys


SCRIPT_NAME = "exporter_main_v2.py"


def export_model(modelname):
    """Exports model passed via `modelname` parameter."""
    
    os.makedirs(os.path.join("exported_models", namespace.model_dir_name), exist_ok=True)
    
    subproc = subprocess.Popen(
        f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --trained_checkpoint_dir="models/{modelname}" --output_directory="exported_models/{modelname}" --input_type=image_tensor --alsologtostderr',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    with open(f"logs/{modelname}_export.log", "w") as logfile:
        logfile.write(f'python {SCRIPT_NAME} --pipeline_config_path="models/{modelname}/pipeline.config" --trained_checkpoint_dir="models/{modelname}" --output_directory="exported_models/{modelname}" --input_type=image_tensor --alsologtostderr\n')
        

    while subproc.poll() is None:
        line = subproc.stderr.readline()
        errors = subproc.stdout.readline()
        with open(f"logs/{modelname}_export.log", "ab") as logfile:
            logfile.write(line)
            logfile.write(errors)
        
        sys.stdout.write(line.decode('utf-8',  errors='ignore'))
        sys.stdout.write(errors.decode('utf-8',  errors='ignore'))


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
    
