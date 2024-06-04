import os
import ast
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from detection_utils.metrics import IoU
from alive_progress import alive_bar
from detection_utils.detections_drawer import draw_intersection


def extract_class_detections(boxes, classname):
    if isinstance(classname, str):
        return [*filter(lambda z: z[-2] == classname, boxes)]
    else:
        if len(boxes) != len(classname):
            raise ValueError("Boxes and classnames have different lengths!")
        return [
            [*filter(lambda z: z[-2] == classname[i], boxes[i])]
            for i in range(len(classname))
        ]


def get_iou(basedir, detection: dict):
    with open(os.path.join(basedir, detection['class'], detection['image'].replace('.jpg', '.json')), 'r') as infile:
        metadata = json.load(infile)
    gtbox = [metadata['bbox']['xmin'], metadata['bbox']['ymin'],
             metadata['bbox']['xmax'], metadata['bbox']['ymax']]
    iou_bbox, iou_val = IoU(gtbox, detection['box'])
    return {
        'image': detection['image'],
        'classname': detection['class'],
        'pbox': detection['box'],
        'gtbox': gtbox,
        'ioubox': iou_bbox,
        'iou': iou_val
    }


def gen_iou(ious, modelname):
    size = len(ious)
    index = 0
    with alive_bar(size) as bar:
        for a in range(size//25 + 1):
            # Create required amount of images.
            f, ax = plt.subplots(5, 5, figsize=(15, 15))
            # Create images grid leaving empty boxes if not divideable by 25.
            f.text(0.5, 0.9, f'Obszary wspólne i IoU dla {modelname}',
                   verticalalignment='center', horizontalalignment='center')
            for i in range(5):
                for j in range(5):
                    if index < size:
                        item = ious[index]
                        draw_intersection(os.path.join(
                            'bricks', item['classname'], item['image']), item['gtbox'], item['pbox'], ax[i, j], (item['ioubox'], item['iou']))
                        bar()
                    else:
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])

                    index += 1

            plt.savefig(os.path.join("workspace", "exported_models", modelname,
                        f"intersect_{a}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(f)
            if index >= size:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelname", help="Name of model dir eg. yolov8n_size_416_bsize_16")
    parser.add_argument("--annotations_basedir", "-ab",
                        help="Base dir for annotations if stored in non-default location", default="bricks", type=str)
    parser.add_argument("--draw", help="Draw IoU and export to image",
                        action=argparse.BooleanOptionalAction)
    namespace = parser.parse_args()

    df = pd.read_csv(os.path.join("workspace", "exported_models",
                     namespace.modelname, "validation_results.csv"))
    df = df[df['class_present']]
    df['boxes'] = [*map(ast.literal_eval, df.boxes.tolist())]
    df["original_class"] = df["image"].apply(lambda z: z[:-9])
    df["class_is_best"] = df["best_class"] == df["original_class"]
    df['boxes'] = extract_class_detections(
        df.boxes.to_list(), df.original_class.to_list())

    flat_boxes = []
    for _, row in df.iterrows():
        detections = [
            {
                'image': row['image'],
                'class': row['original_class'],
                'box': box
            }
            for box in row['boxes']
        ]
        flat_boxes.extend(detections)

    results = [
        *map(lambda z: get_iou(namespace.annotations_basedir, z), flat_boxes)]
    if namespace.draw:
        # Draw intersections if param set
        gen_iou(results, namespace.modelname)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join("workspace", "exported_models",
                   namespace.modelname, "iou_results.csv"), index=False)
