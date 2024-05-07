from shapely.geometry import Polygon


def IoU(bbox_1:list, bbox_2:list):
    """Measures the Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox_1 (list): First bounding box.
        bbox_2 (list): Second bounding box.

    Returns:
        tuple: Two elements tuple containing intersection bounding box and IoU.
    
    >>> IoU([120,80,190,150], [120,90,240,190])
    ((120.0, 90.0, 190.0, 150.0), 0.33070866141732286)
    """
    assert bbox_1[2] >= bbox_1[0]
    assert bbox_1[3] >= bbox_1[1]
    assert bbox_2[2] >= bbox_2[0]
    assert bbox_2[3] >= bbox_2[1]
    
    p1 = Polygon([[bbox_1[0], bbox_1[1]], [bbox_1[2], bbox_1[1]], [bbox_1[2], bbox_1[3]], [bbox_1[0], bbox_1[3]]])
    p2 = Polygon([[bbox_2[0], bbox_2[1]], [bbox_2[2], bbox_2[1]], [bbox_2[2], bbox_2[3]], [bbox_2[0], bbox_2[3]]])
    
    
    union = p1.union(p2)
    intersection = p1.intersection(p2)
    iou = intersection.area / union.area
    if iou > 0:
        return (intersection.bounds, iou)
    return ((0, 0, 0, 0), 0)