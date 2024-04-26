import os
import shutil
import tarfile
import requests

def download_archive(url):
    response = requests.get(url, stream=True)
    if response.ok:
        with open("model_tmp_local", "wb") as archive:
            shutil.copyfileobj(response.raw, archive)
        return True
    return False


def untar_archive(path='model_tmp_local', pre_trained_dir="workspace/pre_trained_models", models_dir="workspace/models"):
    if os.path.exists(path):
        file = tarfile.open(path)
        print("Files:")
        files = file.getnames()
        print(*files, sep='\n')
        file.extractall(os.path.join(pre_trained_dir))
        file.close()
        model_dir = files[0].split('/')[0]
        print(model_dir)
        if not os.path.exists(os.path.join(models_dir, model_dir)):
            os.makedirs(os.path.join(models_dir, model_dir), exist_ok=True)
            shutil.copy2(os.path.join(pre_trained_dir, model_dir, 'pipeline.config'), os.path.join(models_dir, model_dir, 'pipeline.config'))
        return True
    else:
        return False