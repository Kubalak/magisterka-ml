import os
import cv2
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from argparse import ArgumentParser


def box_drawer(row, ax):
    image = cv2.imread(os.path.join("bricks", row["image"][:-9], row["image"]))
    color = 'g' if row['class_is_best'] else 'r'
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    
    if len(row["boxes"]):
        
        boxes = sorted(row["boxes"], key=lambda z: z[-1])
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            rect = patches.Rectangle((box[0], box[1]), w,h, edgecolor=color, facecolor='none', linewidth=1)
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 2, box[-2], verticalalignment='bottom', bbox=dict(facecolor=color, alpha=0.5, boxstyle="square,pad=0.2"), color='w')
        
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("modeldir", help="Model dir name eg. ssd_resnet...")
    parser.add_argument("--datafile", "-df", help="Data file name", type=str, default="validation_results.csv")
    namespace = parser.parse_args()
    
    
    df = pd.read_csv(os.path.join("workspace","exported_models",namespace.modeldir,namespace.datafile))
    df["boxes"] = [*map(ast.literal_eval, df.boxes.to_list())]
    df["original_class"] = df["image"].apply(lambda z: z[:-9])
    df["class_is_best"] = df["best_class"] == df["original_class"]
    
    for a in range(10):
        f, ax = plt.subplots(5, 5, figsize=(15,15))
        for i in range(5):
            for j in range(5):
                box_drawer(df.loc[a * 25 + i * 5 + j], ax[i, j])   
        plt.savefig(os.path.join("workspace","exported_models",namespace.modeldir,f"detections_{a}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{(a+1):2d}/10", end="\r")
    
    print()
    
    print("Images written to", namespace.modeldir)