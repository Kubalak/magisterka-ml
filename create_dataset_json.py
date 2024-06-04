import tensorflow as tf
import os
import re
import time
import json
import argparse


if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basedir", "-bdir", help="Base dir to search for files", type=str, default="data")
    parser.add_argument("--output_dir", "-od",
                        help="Output dir to write files to (in workspace dir)", type=str, default="data")

    namespace = parser.parse_args()

    files = os.listdir(os.path.join("workspace", namespace.basedir))
    files = [*filter(lambda z: z.endswith('.record'), files)]

    regex = re.compile("^\\d+")

    with open("dataset.json", "r") as infile:
        dataset = json.load(infile)

    dataset["train"] = [*map(lambda z: z.split("/")[-1], dataset["train"])]
    dataset["train"] = [
        *map(lambda z: '.'.join([*z.split(".")[:-1], 'record']), dataset["train"])]

    dataset["eval"] = [*map(lambda z: z.split("/")[-1], dataset["eval"])]
    dataset["eval"] = [
        *map(lambda z: '.'.join([*z.split(".")[:-1], 'record']), dataset["eval"])]

    train_records = tf.data.TFRecordDataset(
        [*map(lambda z: os.path.join('workspace', namespace.basedir, z), dataset["train"])])
    eval_records = tf.data.TFRecordDataset(
        [*map(lambda z: os.path.join('workspace', namespace.basedir, z), dataset["eval"])])

    if not os.path.exists(os.path.join("workspace", namespace.output_dir)):
        os.makedirs(os.path.join(
            "workspace", namespace.output_dir), exist_ok=True)
    elif os.path.isfile(os.path.join("workspace", namespace.output_dir)):
        print("Target direcory is file!")
        exit(-1)

    print("Writing data to train dataset")

    with tf.io.TFRecordWriter(os.path.join("workspace", namespace.output_dir, "train.tfrecords")) as writer:
        for record in train_records.take(len(dataset['train'])):
            tf_record = tf.train.Example()
            tf_record.ParseFromString(record.numpy())
            writer.write(tf_record.SerializeToString())

    print("Writing data to eval dataset")

    with tf.io.TFRecordWriter(os.path.join("workspace", namespace.output_dir, "eval.tfrecords")) as writer:
        for record in eval_records.take(len(dataset['eval'])):
            tf_record = tf.train.Example()
            tf_record.ParseFromString(record.numpy())
            writer.write(tf_record.SerializeToString())

    stop = time.time()

    print(f"Done in {stop-start} s")
