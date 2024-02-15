import cv2
import numpy as np
import json, os
import tensorflow as tf
import re
import utils

def prepare(filename):
    img = cv2.imread(filename,0)
    height,width = img.shape
    return [[img.item(j,i) != 0 for j in range(height)] for i in range(width)]

def get_bounds(pixels):
    width = len(pixels)
    height = len(pixels[0])
    annotations = {
        "xmin": width,
        "xmax": 0,
        "ymin": height,
        "ymax": 0
    }
    for i in range(width):
        for j in range(height):
            if pixels[i][j] and i < annotations["xmin"]:
                annotations["xmin"] = i
            elif pixels[i][j] and i > annotations["xmax"]:
                annotations["xmax"] = i
            if pixels[i][j] and j < annotations["ymin"]:
                annotations["ymin"] = j
            elif pixels[i][j] and j > annotations["ymax"]:
                annotations["ymax"] = j
    annotations["xmin"] -= 1
    annotations["xmax"] += 1
    annotations["ymin"] -= 1
    annotations["ymax"] += 1
    return annotations

def annotate(filename:str, class_name, class_id):
    pixels = prepare(filename)
    width = len(pixels)
    height = len(pixels[0])
    info = get_bounds(pixels)
    expr = re.compile("\\.[^\\.]*$")
    f_format = expr.search(filename).group(0)
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/filename': utils.bytes_feature(filename.split('/')[-1].encode()),
        'image/source_id': utils.bytes_feature(filename.encode()),
        'image/encoded': utils.bytes_feature(encoded_image),
        'image/format': utils.bytes_feature(f_format[1:].encode()),
        'image/object/bbox/xmin': utils.float_list_feature([info["xmin"] / width]),
        'image/object/bbox/xmax': utils.float_list_feature([info["xmax"] / width]),
        'image/object/bbox/ymin': utils.float_list_feature([info["ymin"] / height]),
        'image/object/bbox/ymax': utils.float_list_feature([info["ymax"] / height]),
        'image/object/class/text': utils.bytes_list_feature([class_name.encode()]),
        'image/object/class/label': utils.int64_list_feature([class_id]),
    }))

def annotate_dir(dirname):
    expr = re.compile("^\\d+")
    class_id = int(expr.search(dirname).group(0))
    class_name = dirname
    files = os.listdir(os.path.join("bricks", dirname))
    files = filter(lambda z: z != "annotations.json", files)
    records = [*map(lambda z: annotate(os.path.join("bricks", dirname, z), class_name, class_id), files)]
    for index, record in enumerate(records):
        with tf.io.TFRecordWriter(os.path.join("workspace", "data", f"{class_name}-{index}.record")) as writer:
            writer.write(record.SerializeToString())
    return {
        "id": class_id,
        "name": class_name
    }

if __name__ == "__main__":
    basedirs = os.listdir("bricks")
    annotations = []
    for index,base in enumerate(basedirs):
        print(f"[{index + 1}/{len(basedirs)}] Working on: {base}")
        annotations.append(annotate_dir(base))
        
        with open(os.path.join("workspace", "data", "label_map.pbtxt"), "w") as f:
            f.writelines(map(lambda z: json.dumps(z), annotations))
        
        print("", end="\033[F\033[K")