import tensorflow as tf
import json


def extract_filenames(filename):
    dataset = tf.data.TFRecordDataset(filename)
    filenames = []
    for item in dataset:
        record = tf.train.Example()
        record.ParseFromString(item.numpy())
        
        filename = record.features.feature['image/source_id'].bytes_list.value[0].decode("utf-8")
        filenames.append(filename)
    return filenames


if __name__ == "__main__":
    train = extract_filenames("workspace/data/train.tfrecords")
    eval = extract_filenames("workspace/data/eval.tfrecords")
    
    with open("dataset.json", "w") as o:
        json.dump({"train": train, "eval": eval}, o, indent=2)