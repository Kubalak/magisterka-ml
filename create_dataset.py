import tensorflow as tf
import os
import re
import time
import argparse
import random


if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basedir", "-bdir", help="Base dir to search for files", type=str, default="data")
    parser.add_argument("--filenum", "-fn",
                        help="Number of files per class", type=int, default=10)
    parser.add_argument(
        "--ratio", "-rt", help="How much data to training dataset [0-1]", type=float, default=0.8)
    parser.add_argument("--output_dir", "-od",
                        help="Output dir to write files to (in workspace dir)", type=str, default="data")

    namespace = parser.parse_args()

    files = os.listdir(os.path.join("workspace", namespace.basedir))
    files = [*filter(lambda z: z.endswith('.record'), files)]

    regex = re.compile("^\\d+")
    classes = {}

    for file in files:
        classes[regex.search(file).group(0)] = []

    for file in files:
        classes[regex.search(file).group(0)].append(file)

    for class_name in classes:
        classes[class_name] = classes[class_name][:namespace.filenum]

    train_ds = []
    eval_ds = []

    for class_id in classes:
        num_choices = len(classes[class_id])
        choices = [*range(num_choices)]
        _train = random.sample(choices, int(namespace.ratio * num_choices))
        _eval = [*filter(lambda z: z not in _train, choices)]
        train_ds.extend(map(lambda z: classes[class_id][z], _train))
        eval_ds.extend(map(lambda z: classes[class_id][z], _eval))

    train_records = tf.data.TFRecordDataset(
        [*map(lambda z: os.path.join('workspace', namespace.basedir, z), train_ds)])
    eval_records = tf.data.TFRecordDataset(
        [*map(lambda z: os.path.join('workspace', namespace.basedir, z), eval_ds)])

    if not os.path.exists(os.path.join("workspace", namespace.output_dir)):
        os.makedirs(os.path.join(
            "workspace", namespace.output_dir), exist_ok=True)
    elif os.path.isfile(os.path.join("workspace", namespace.output_dir)):
        print("Target direcory is file!")
        exit(-1)

    print("Writing data to train dataset")

    with tf.io.TFRecordWriter(os.path.join("workspace", namespace.output_dir, "train.tfrecords")) as writer:
        for record in train_records.take(len(train_ds)):
            tf_record = tf.train.Example()
            tf_record.ParseFromString(record.numpy())
            writer.write(tf_record.SerializeToString())

    print("Writing data to eval dataset")

    with tf.io.TFRecordWriter(os.path.join("workspace", namespace.output_dir, "eval.tfrecords")) as writer:
        for record in eval_records.take(len(eval_ds)):
            tf_record = tf.train.Example()
            tf_record.ParseFromString(record.numpy())
            writer.write(tf_record.SerializeToString())

    stop = time.time()

    print(f"Done in {stop-start} s")
