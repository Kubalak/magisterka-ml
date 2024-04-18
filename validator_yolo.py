import os
import pandas as pd
from ultralytics import YOLO
from yolo.yolo_detect import detect
from argparse import ArgumentParser
from alive_progress import alive_bar

def check_detection(row, model):
    result = detect(row["image"], model)
    classes = []
    best_class = 'N/A'
    best = -1
    class_score = 'N/A'
    
    for box in result["boxes"]:
        classes.append(box[-2])
        if box[-2] == row['class']:
            class_score = box[-1]
        if box[-1] > best:
            best = box[-1]
            best_class = box[-2]
    
    result["best"] = best if best > 0 else "N/A"
    result["best_class"] = best_class
    result["class_present"] = row["class"] in classes
    result["class_score"] = class_score
    
    return result
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", help="Path to model")
    parser.add_argument("--threshold", "-th", help="Detection threshold", type=float, default=0.2)
    
    namespace = parser.parse_args()
    model = YOLO(namespace.model)
    
    validations = pd.read_csv("validation.csv")
    
    results = []
    
    with alive_bar(int(validations["class"].count())) as bar:
        for _,item in validations.iterrows():
            results.append(check_detection(item, model))
            bar()
    
    results = pd.DataFrame(results)
    results.to_csv(os.path.join("yolo", "validation_results.csv"), index=False)
    
    