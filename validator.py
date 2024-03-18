import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from object_detection.utils import label_map_util
from alive_progress import alive_bar


def ExtractBBoxes(bboxes, bclasses, bscores, threshold, im_width, im_height, category_index):
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
    return (bbox, class_labels)

# @Matheus Correia's code but modified

def detection(image_path, width, height, model_fn, threshold, category_index):
    # Pre-processing image.
    image = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
    image = tf.image.resize(image, (width,height))
    im_height, im_width, _ = image.shape
    # Model expects tf.uint8 tensor, but image is read as tf.float32.
    input_tensor = np.expand_dims(image, 0)
    detections = model_fn(input_tensor)

    bboxes = detections['detection_boxes'][0].numpy()
    bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
    bscores = detections['detection_scores'][0].numpy()
    return ExtractBBoxes(bboxes, bclasses, bscores, threshold, im_width, im_height, category_index)

def evaluate_detection(image_path, width, height, model_fn, threshold, category_index, actual_class):
    # boxes = [[145, 172, 495, 461, '6632 lever 3M', 0.2697998285293579]]
    # classes = ['6632 lever 3M']
    start = datetime.now()
    boxes, classes = detection(image_path, width, height, model_fn, threshold, category_index)
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

    parser.add_argument("--model_path", "-mp", help="Path to model eg. efficientdet_d0_coco17_tpu-32", type=str, required=True)
    parser.add_argument("--labels_path", "-lp", help="Path to labels map", type=str, default="workspace/data/label_map.pbtxt")
    parser.add_argument("--threshold", "-th", help="Detection threshold", type=float, default=0.2)
    parser.add_argument("--im_width", "-iw", help="Image width", type=int, default=400)
    parser.add_argument("--im_height", "-ih", help="Image height", type=int, default=400)
    
    namespace = parser.parse_args()
    
    category_index = label_map_util.create_category_index_from_labelmap(namespace.labels_path, use_display_name=True)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    model_fn = tf.saved_model.load(os.path.join("workspace", "exported_models", namespace.model_path, "saved_model"))

    validations = pd.read_csv("validation.csv")
    
    results = []
    
    with alive_bar(int(validations["class"].count())) as bar:
        for _,item in validations.iterrows():
            results.append(evaluate_detection(item["image"], namespace.im_width, namespace.im_height, model_fn, namespace.threshold, category_index, item["class"]))
            bar()

    results = pd.DataFrame(results)

    results.to_csv(os.path.join("workspace", "exported_models", namespace.model_path, "validation_results.csv"), index=False)
    
