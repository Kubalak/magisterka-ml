import os
import json
import time
import utils
import tensorflow as tf
from multiprocessing import Pool, cpu_count

states = ['|', '/', '-', '\\']


def to_tfrecord(filename):
    """Reads a `filename` and associated `*.json` file with annotations.
    Returns:
    tf.train.Example -- data in TFRecord format.
    """
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()
    meta_filename = '.'.join([*filename.split('.')[:-1], 'json'])
    with open(meta_filename, 'r') as infile:
        obj = json.load(infile)

    if obj['bbox']["xmin"] < 0:
        raise RuntimeError("xmin < 0")
    if obj['bbox']["xmax"] > obj["width"]:
        raise RuntimeError("xmax > width")
    if obj['bbox']["ymin"] < 0:
        raise RuntimeError("ymin < 0")
    if obj['bbox']["ymax"] > obj["height"]:
        raise RuntimeError("ymax > height")
    height = obj['height']
    width = obj['width']
    return ('.'.join(obj['filename'].split('.')[:-1]), tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/filename': utils.bytes_feature(obj['filename'].encode()),
        'image/source_id': utils.bytes_feature(obj['source_id'].encode()),
        'image/encoded': utils.bytes_feature(encoded_image),
        'image/format': utils.bytes_feature(obj['format'].encode()),
        'image/object/bbox/xmin': utils.float_list_feature([obj['bbox']["xmin"] / width]),
        'image/object/bbox/xmax': utils.float_list_feature([obj['bbox']["xmax"] / width]),
        'image/object/bbox/ymin': utils.float_list_feature([obj['bbox']["ymin"] / height]),
        'image/object/bbox/ymax': utils.float_list_feature([obj['bbox']["ymax"] / height]),
        'image/object/class/text': utils.bytes_list_feature([obj['class/text'].encode()]),
        'image/object/class/label': utils.int64_list_feature([obj['class/label']]),
    })))


def tf_from_dir(dirname):
    """Scans directory and converts all JPG files into `*.record` files."""
    files = filter(lambda z: z.endswith('.jpg'),
                   os.listdir(os.path.join('bricks', dirname)))
    records = [
        *map(lambda z: to_tfrecord(os.path.join('bricks', dirname, z)), files)]
    for id, record in records:
        with tf.io.TFRecordWriter(os.path.join("workspace", "tfrecords", f"{id}.record")) as writer:
            writer.write(record.SerializeToString())


if __name__ == "__main__":
    start = time.time()
    directories = [
        *filter(lambda z: os.path.isdir(os.path.join("bricks", z)), os.listdir("bricks"))]

    with Pool(cpu_count()) as pool:
        pool.map(tf_from_dir, directories)

    stop = time.time()
    print(f"Job took {stop-start}s to complete")
