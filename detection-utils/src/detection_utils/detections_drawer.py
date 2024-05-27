"""
Module providing utilities for detections drawing.

Jakub Jach &copy; 2024
"""
import os
import cv2
import ast
import pandas as pd
import matplotlib.axes
from .metrics import IoU
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from argparse import ArgumentParser
import matplotlib.patches as patches
from typing import List

def box_drawer(image:str|bytes, classname:str, boxes:List[int], ax:matplotlib.axes.Axes):
    """Draws a detection on image provided.

    Args:
        image (str|bytes): Path to image or image encoded as bytes.
        classname (str): Name of valid class.
        boxes (list): Detection boxes returned in UniversalDetector dict.
        ax (matplotlib.axes.Axes): Axes on which to draw detections
    """
    if type(image) is bytes:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image)
    
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    
    if len(boxes):
        boxes = sorted(boxes, key=lambda z: z[-1]) 
        #Ensures prediction with highest score is on top.
        for box in boxes:
            color = 'g' if classname == box[-2] else 'r'
            w = box[2] - box[0]
            h = box[3] - box[1]
            rect = patches.Rectangle((box[0], box[1]), w,h, edgecolor=color, facecolor='none', linewidth=1)
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 2, box[-2], verticalalignment='bottom', bbox=dict(facecolor=color, alpha=0.5, boxstyle="square,pad=0.2"), color='w')


def draw_intersection(image:str|bytes, gtbox:List[int], pbox:List[int], ax:matplotlib.axes.Axes, det_iou:tuple=None):
    """Draws an intersection of real bounding box and predicted bounding box on image and puts it onto axis.

    Args:
        image (str|bytes): Path to image or bytes containing image content.
        gtbox (list): Ground truth bounding box. Requires format [xmin, ymin, xmax, ymax]
        pbox (list): Predicted bounding box. Accepts box returned by UniversalDetector object.
        ax (matplotlib.axes.Axes): Axes on which image with detection and intersection will be drawn.
        det_iout (tuple)=None: Detected IoU (returned by IoU function). If not `None` it's used to draw intersection.
    """
    if type(image) is bytes:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    wgt = gtbox[2] - gtbox[0]
    hgt = gtbox[3] - gtbox[1]
    
    wp = pbox[2] - pbox[0]
    hp = pbox[3] - pbox[1]
    gt = patches.Rectangle((gtbox[0], gtbox[1]), wgt, hgt,edgecolor='g', facecolor='none', linewidth=1)
    ax.add_patch(gt)
    
    p = patches.Rectangle((pbox[0], pbox[1]), wp, hp, edgecolor='m', facecolor='none', linewidth=1)
    ax.add_patch(p)
    ibox, iou = None, None
    if det_iou is not None:
        ibox, iou = det_iou
    else:
        ibox, iou = IoU(gtbox, pbox)
    if iou > 0:
        wi = ibox[2] - ibox[0]
        hi = ibox[3] - ibox[1]
        inter = patches.Rectangle((ibox[0], ibox[1]), wi, hi, facecolor='c', alpha=0.2)
        ax.add_patch(inter)
    
    ax.text(gtbox[0] + wgt / 2, gtbox[3], f"IoU: {iou:.2f}", horizontalalignment='center', verticalalignment='bottom', color='w')
 

def generate_bars(modelpath:str, df:pd.DataFrame=None, close=False):
    results = None
    if df is None:
        results = pd.read_csv(os.path.join(modelpath, "validation_results.csv"))
    else:
        results = df.copy()
    results["original_class"] = results["image"].apply(lambda z: z[:-9])
    results["class_is_best"] = results["best_class"] == results["original_class"]
    agg_dict = {
        "class_present": lambda z: z.sum(),
        "class_is_best": lambda z: z.sum()
    }
    grouped = results.groupby("original_class", as_index=False).agg(agg_dict)

    figure = plt.figure(figsize=(12,24))
    modelname = modelpath.split('/')[-1].split('\\')[-1]
    plt.title(f"{modelname} - Liczba detekcji dla klasy")
    bar1 = plt.barh(grouped['original_class'], grouped['class_present'],  label="Oryginalna klasa na liście wykrytych klas")
    bar2 = plt.barh(grouped['original_class'], grouped['class_is_best'], label="Oryginalna klasa ma najwyższy wynik")
    plt.legend(handles=[bar1,bar2])
    # figure.autofmt_xdate(rotation=90, ha='center')
    plt.grid(axis='x')
    plt.savefig(os.path.join(modelpath, "figure.png"), pad_inches=0.1, dpi=300, bbox_inches='tight')
    if close:
        plt.close(figure)


def create_detections_grid(modelpath:str, datafile:str):
    df = pd.read_csv(os.path.join(modelpath, datafile))
    df["boxes"] = [*map(ast.literal_eval, df.boxes.to_list())]
    df["original_class"] = df["image"].apply(lambda z: z[:-9])
    df["class_is_best"] = df["best_class"] == df["original_class"]
    
    with alive_bar(int(df['image'].count())) as bar:
        for a in range(10):
            f, ax = plt.subplots(5, 5, figsize=(15,15))
            for i in range(5):
                for j in range(5):
                    row = df.loc[a * 25 + i * 5 + j]
                    box_drawer(os.path.join("bricks", row["image"][:-9], row["image"]), row["image"][:-9], row["boxes"], ax[i, j])
                    bar()   
            plt.savefig(os.path.join(modelpath,f"detections_{a}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(f)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("modeldir", help="Model dir name eg. workspace/exported_models/ssd_resnet...")
    parser.add_argument("--datafile", "-df", help="Data file name", type=str, default="validation_results.csv")
    namespace = parser.parse_args()
    
    create_detections_grid(namespace.modeldir, namespace.datafile)
    
    print("Images written to", namespace.modeldir)