import json, os
from alive_progress import alive_bar

join = os.path.join

def get_yolo(filename:str):
    with open(filename, 'r') as i:
        annotations = json.load(i)
    class_id = annotations['class/label'] - 1
    x_center = (annotations['bbox']["xmin"] + annotations['bbox']["xmax"]) / 2 / annotations["width"]
    y_center = (annotations['bbox']["ymin"] + annotations['bbox']["ymax"]) / 2 / annotations["height"]
    w = (annotations['bbox']["xmax"] - annotations['bbox']["xmin"]) / annotations["width"]
    h = (annotations['bbox']["ymax"] - annotations['bbox']["ymin"]) / annotations["height"]
    filename = filename.split('/')[-1]
    filename = ".".join(filename.split(".")[:-1])
    return (f"{filename}.txt", f"{class_id} {x_center} {y_center} {w} {h}")
    
    

if __name__ == "__main__":
    classes = [*filter(lambda z: os.path.isdir(join("bricks", z)), os.listdir("bricks"))]
    with alive_bar(len(classes)) as bar:
        for class_name in classes:
            files = [*filter(lambda z: z.endswith(".json"), os.listdir(join("bricks", class_name)))]
            results = [*map(get_yolo, map(lambda z: join("bricks", class_name, z), files))]
            for file, label in results:
                with open(join("yolo", "dataset", "labels", file), "w") as o:
                    o.write(label)
            bar()
        
    