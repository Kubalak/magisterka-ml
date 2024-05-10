"""
Module to provide universal API for both YOLO and TensorFlow models.

Jakub Jach &copy; 2024
"""
import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
from .yol.detect import detect
from timeit import default_timer as timer
from object_detection.utils import label_map_util


class UniversalObjectDetector:
    """Class made to provide universal interface for YOLO and TensorFlow object detection models."""
    
    
    def __init__(self, modelpath, category_index=None):
        """Initializes object with appropriate model - `YOLO` or `Tensorflow`.
        You need to pass `category_index` path when using `TensorFlow` model.

        Args:
            modelpath (str): Path to model directory which contains `weights` YOLO model or `saved_model` for TensorFlow`.
            category_index (str): Path to label map. Used for TensorFlow models.
        Raises:
            FileNotFoundError: If directory does not exist or does not contain required file.
            RuntimeError: If directory contains both YOLO and TensorFlow models.
        """
        modeldir = os.path.join( modelpath)
        if not os.path.exists(modeldir):
            raise FileNotFoundError(f"Model {modelpath} not found in exported models.")
        modeltype = None
        yolomp = os.path.join(modelpath, "weights")
        tfmp = os.path.join(modelpath, "saved_model")
        if os.path.exists(yolomp) and os.path.isdir(yolomp):
            if not os.path.exists(os.path.join(modelpath, "weights", "best.pt")):
                raise FileNotFoundError("No best.pt file found in yolo model dir.")
            modeltype = 'yolo'
        if os.path.exists(tfmp) and os.path.isdir(tfmp):
            if modeltype is not None:
                raise RuntimeError("Model directory contains both TensorFlow and YOLO models.")
            modeltype = 'tf'
        if modeltype is None:
            raise FileNotFoundError("Selected model does not contain weights or checkpoint directory.")
        
        if modeltype == 'tf':
            self.__category_index = label_map_util.create_category_index_from_labelmap(category_index, use_display_name=True)
            self.__model = tf.saved_model.load(tfmp)
        else:
            self.__model = YOLO(os.path.join(modelpath, "weights", "best.pt"))
        
        self.__modeltype = modeltype
    
    
    def detect_objects(self, file:str|bytes, threshold=.5):
        """Detects LEGO bricks on image passed in filename param

        Args:
            file (str|bytes): Name of image file or its conent.
            threshold (float): Detection threshold. Detections with confidence lower than this will not be returned.

        Returns:
            dict: Dictionary with detections.
        """
        if not isinstance(file, str) and not isinstance(file, bytes):
            raise TypeError("file allowed only to be str or bytes!")
        
        if self.__modeltype == 'tf':
            return self.__detect_tf(file, threshold)
        return self.__detection_yolo(file, threshold)
    
    
    def raw_detection(self, file:str|bytes, **kwargs):
        """Raw detection of loaded model.

        Args:
            file (str|bytes): Path to image file or content.

        Returns:
            Response of `YOLO` or `TensorFlow` model.
        """
        if not isinstance(file, str) and not isinstance(file, bytes):
            raise TypeError("Invalid filename argument type! Allowed only str and bytes.")
        raw_content = file
        if type(file) is str:
            with open(file, 'rb') as infile:
                raw_content = infile.read()
        if self.__modeltype == 'tf':
            image = tf.image.decode_image(raw_content, channels=3)
            input_tensor = np.expand_dims(image, 0)
            return self.__model(input_tensor, **kwargs)
        else:
            return self.__model.predict(Image.open(io.BytesIO(raw_content)), **kwargs)
             
    
    def __detection_yolo(self, filename, th):
        return detect(filename, self.__model, th)
    
    
    def __ExtractBBoxes(self, bboxes, bclasses, bscores, threshold, im_width, im_height, category_index):
        bbox = []
        class_labels = []
        for idx in range(len(bboxes)):
            if bscores[idx] >= threshold:
                y_min = int(bboxes[idx][0] * im_height)
                x_min = int(bboxes[idx][1] * im_width)
                y_max = int(bboxes[idx][2] * im_height)
                x_max = int(bboxes[idx][3] * im_width)
                class_label = category_index[int(bclasses[idx])]['name']
                class_labels.append(class_label)
                bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
        return bbox
    
    
    def __detect_tf(self, filename, th):
        # Pre-processing image.
        start = timer()
        data = filename
        if type(filename) is str:
            with open(filename, 'rb') as infile:
                data = infile.read()
        image = tf.image.decode_image(data, channels=3)
        
        im_height, im_width, _ = image.shape
        # Model expects tf.uint8 tensor, but image is read as tf.float32.
        input_tensor = np.expand_dims(image, 0)
        preproc = timer()
        detections = self.__model(input_tensor)
        det = timer()

        bboxes = detections['detection_boxes'][0].numpy()
        bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
        bscores = detections['detection_scores'][0].numpy()
        
        boxes = self.__ExtractBBoxes(bboxes, bclasses, bscores, th, im_width, im_height, self.__category_index)
        
        post = timer()
        results = {}
        
        if type(filename) is str:
            filename = filename.split("/")[-1]
            filename = filename.split("\\")[-1]
            results["filename"] = filename
        
        results['time'] = {
            'preprocess': (preproc-start)*1000,
            'inference': (det-preproc)*1000,
            'postprocess': (post-det)*1000
        }
        results['boxes'] = boxes
        
        return results