from ultralytics import YOLO
from PIL import Image
from argparse import ArgumentParser

def detect(filename:str, model:YOLO, threshold=0.0):
    img = Image.open(filename)
    results = model.predict(img, agnostic_nms=True, verbose=False)
    filename = filename.split("/")[-1]
    filename = filename.split("\\")[-1]
    
    detecions = {
        "image": filename,
        "time": None,
        "boxes": []
    }
    
    if len(results) > 1:
        print("Warn multiple results found")
    
    result = results[0]
    
    detecions["time"] = result.speed
    
    for i in range(len(result.boxes)):
        if result.boxes.conf[i].item() < threshold:
            continue
        class_name = model.names[int(result.boxes.cls[i])]
        bbox = result.boxes.xyxy[i]
        
        detecions["boxes"].append(
            [int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item()),  class_name, result.boxes.conf[i].item()]
        )
    return detecions

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", help="Path to trained model", required=True)
    parser.add_argument("--image", "-img", help="Path to image", required=True)
    parser.add_argument("--labels", "-lab", help="Path to labels mapping file", required=True)
    
    namespace = parser.parse_args()
    
    model = YOLO(namespace.model)
    image = Image.open(namespace.image)
    
    results = model.predict(image)
    for result in results:
        img_name = result.path.split("\\")[-1]
        img_name = img_name.split("/")[-1]
        result.save(f"detect-{img_name}")
    
    