import argparse
import pandas as pd
import tensorflow as tf
from datetime import datetime
from alive_progress import alive_bar
from detection_utils.universal_detection_api import UniversalObjectDetector
from detection_utils.detections_drawer import generate_bars


def evaluate_detection(image_path, detector:UniversalObjectDetector, threshold:float, actual_class:str):
    """Evaluates detection on selected image.

    Args:
        image_path (str|bytes): Path to image file or bytes.
        detector (UniversalObjectDetector): Detector object.
        threshold (float): Detection threshold.
        actual_class (str): Actual class on the image.

    Returns:
        dict: Detections dict that can be used to create DataFrame.
    """
    # boxes = [[145, 172, 495, 461, '6632 lever 3M', 0.2697998285293579]]
    # classes = ['6632 lever 3M']
    start = datetime.now()
    detections = detector.detect_objects(image_path, threshold)#detection(image_path, width, height, model_fn, threshold, category_index)
    boxes = detections['boxes']
    classes = [*map(lambda z: z[-2], boxes)]
    stop = datetime.now()
    class_present = actual_class in classes

    class_score = "N/A"

    if class_present:
        class_score = [*filter(lambda z: z[-2] == actual_class, boxes)][0][-1]
    
    best_box = boxes[0][-1] if len(classes) > 0 else "N/A"
    best_class = boxes[0][-2] if len(classes) > 0 else "N/A"

    for box in boxes:
        if box[-1] > best_box:
            best_box = box[-1]
            best_class = box[-2]

    return {
        'image': image_path.split("/")[-1],
        'time': (stop-start).total_seconds() * 1000,
        'boxes': boxes,
        'best': best_box,
        'best_class': best_class,
        'class_present': class_present,
        'class_score': class_score if class_present else "N/A"
    }



if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", help="Path to model eg. efficientdet_d0_coco17_tpu-32", type=str)
    parser.add_argument("--labels_path", "-lp", help="Path to labels map", type=str, default="workspace/data/label_map.pbtxt")
    parser.add_argument("--threshold", "-th", help="Detection threshold", type=float, default=0.2)
    parser.add_argument("--genbars", help="Generate bars for detections", action=argparse.BooleanOptionalAction)
    
    namespace = parser.parse_args()
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    detector = UniversalObjectDetector(os.path.join("workspace", "exported_models", namespace.model_path), namespace.labels_path)

    validations = pd.read_csv("validation.csv")
    
    results = []
    
    with alive_bar(int(validations["class"].count())) as bar:
        for _,item in validations.iterrows():
            results.append(evaluate_detection(item["image"], detector, namespace.threshold, item["class"]))
            bar()

    results = pd.DataFrame(results)
    if namespace.genbars:
        print("Generating bars plot.")
        generate_bars(os.path.join("workspace", "exported_models", namespace.model_path), results, True)

    results.to_csv(os.path.join("workspace", "exported_models", namespace.model_path, "validation_results.csv"), index=False)