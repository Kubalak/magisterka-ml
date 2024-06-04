import os
import json
import cv2
import time
from alive_progress import alive_bar


def prepare(filename):
    """Loads the filename and prepares True/False matrix representing pixels.

    Args:
        filename (str): Path to image.

    Returns:
        tuple(MatLike,list[list[bool]]): Two elements tuple with loaded image and `True`/`False` matrix.
    """
    img = cv2.imread(filename, 0)
    height, width = img.shape
    return (img, [[img.item(j, i) != 0 for j in range(height)] for i in range(width)])


def get_bounds(pixels):
    """Creates a bounding box for passed matrix.

    Args:
        pixels (list[list[bool]]): Matrix representing pixels.

    Returns:
        dict[str,int]: Dict containing `xmin`, `ymin`, `xmax` and `ymax` points of bounding box.
    """
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

    if annotations["xmin"] > 0:
        annotations["xmin"] -= 1
    if annotations["xmax"] < width:
        annotations["xmax"] += 1
    if annotations["ymin"] > 0:
        annotations["ymin"] -= 1
    if annotations["ymax"] < height:
        annotations["ymax"] += 1
    return annotations


def annotate(filename: str, class_name, class_id):
    """Annotates given filename.

    Args:
        filename (str): Path to file
        class_name (str): Name of the class.
        class_id (int): Class id starting from 1.

    Returns:
        tuple: Two element tuple with image and annotation.
    """
    image, pixels = prepare(filename)
    width = len(pixels)
    height = len(pixels[0])
    info = get_bounds(pixels)
    splitted_name = filename.split('.')[:-1]
    filename = '.'.join([*splitted_name, 'jpg'])
    return (image, {
        'height': height,
        'width': width,
        'filename': filename.split('/')[-1],
        'source_id': filename,
        'format': 'jpg',
        'bbox': info,
        'class/text': class_name,
        'class/label': class_id,
    })


def mark(directories):
    """Creates annotation for directories and converts images to JPG.

    Args:
        directories (list[str]): List of directories containing images.
    """
    files = {}
    meta = []
    index = 0
    with alive_bar(len(directories)) as bar:
        for directory in directories:
            files[directory] = [
                *filter(lambda z: z.endswith('.jpg'), os.listdir(os.path.join('bricks', directory)))]
            # expr = re.compile("^\\d+")
            # class_id = int(expr.search(dirname).group(0))
            index += 1
            for file in files[directory]:
                image, info = annotate(os.path.join(
                    'bricks', directory, file), directory, index)
                cv2.imwrite(os.path.join('bricks-jpg', directory,
                            info['filename']), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                with open(os.path.join('bricks-jpg', directory, '.'.join([*file.split('.')[:-1], 'json'])), 'w') as file:
                    json.dump(info, file, indent=2)

            meta.append({'id': index, 'name': directory})
            bar()

    with open("labels.pbtxt", "w") as file:
        for item in meta:
            file.write(
                f"item {'{'}\n  id: {item['id']}\n  name: \'{item['name']}\'\n{'}'}\n")
    print()


if __name__ == "__main__":
    start = time.time()
    directories = os.listdir("bricks")
    mark(directories)
    stop = time.time()
    print(f"Job took {stop-start}s to complete")
