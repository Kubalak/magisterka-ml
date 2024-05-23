# Testing usage of object detection to detect LEGO brick on images

## Useful links

- [LEGO dataset](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)
- [Neptune AI Post](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api)
- [Ultralytics](https://www.ultralytics.com/)

### Reqirements

- Python 3.9+ (tested on 3.10.13)
- TensorFlow 2.10(+? tested on 2.10)
- PyTorch 2.2.2
- Ultralytics 8.2.2
- [Protoc](https://github.com/protocolbuffers/protobuf/releases) to build object-detection source
- Pandas 
- Pillow 9.4 (doesn't work with 10+)
- PyYAML 5.4.1 (install with conda pip package is broken)
- OpenCV-python
- Matplotlib
- Shapely
- Protobuf 3.20.3 (doesn't work with other versions)
- CUDA capable GPU to speed up calculations

### Installation

First clone the repo to local folder.

`git clone https://github.com/Kubalak/magisterka-ml.git`

Then move into the cloned repo.

`cd magisterka-ml`

Next step is to clone TensorFlow object-detection utils.

`git clone https://github.com/tensorflow/models.git`

Then you should build object detection library using protoc.

`protoc models/research/object_detection/protos/*.proto --python_out=.`

Alternatively you can just use `models.7z` inside `models` folder.
`7z x models/models.7z -o models`

In the main directory go to detection-utils:

`cd detection-utils`

And install with `pip install .`

