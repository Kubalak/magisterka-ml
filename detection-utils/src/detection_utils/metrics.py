import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.metrics import classification_report


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

    
def _row_support(row, class_name, require_max):
    if require_max:
        return (class_name, row['best_class'])
    return (class_name, class_name if row['class_present'] else row['best_class'])


def matrix_from_df(df:pd.DataFrame, require_class_max = True):
    """Create classes matrix to be used by scikit classification_report.
    
    Args:
        df (pd.DataFrame): Pandas DataFrame to be used.
        require_class_max (bool): Require class to have max value among detection boxes.
    
    Returns:
        np.ndarray: NumPy array containing classes [y_true, y_pred].
    """
    classes = df['original_class'].unique()
    classes_dict = {
        classes[i] : i
        for i in range(len(classes))
    }
    classes_dict["-"] = 51 # Include not classified in classes.
    results = [*map(lambda x: _row_support(x[1], x[1]['original_class'], require_class_max), df.iterrows())]
    results = np.array([*map(lambda x: [classes_dict[x[0]], classes_dict[x[1]]], results)], dtype=np.int32)
    return np.array([results[:, 0], results[:, 1]])
    

def class_report(y_true:np.ndarray, y_pred:np.ndarray) -> dict:
    """Generates class report dict from class predictions.
    
    Args:
        y_true (np.ndarray): An array containing true classes.
        y_pred (np.ndarray): An array containing predicted classes.
    Returns:
        dict: Dictionary returned by scikit classification_report.
    """
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)
       

def precision(report:dict):
    """Returns precision from report generated by `class_report`"""
    return {
        'weighted': report['weighted avg']['precision'],
        'macro': report['macro avg']['precision']
    }


def recall(report:dict):
    """Returns recall from report generated by `class_report`"""
    return {
        'weighted': report['weighted avg']['recall'],
        'macro': report['macro avg']['recall']
    }


def generate_report(pr:dict, rc:dict):
    """Generates report from precision and recall.
    
    Args:
        pr (dict): Precision dict returned by `precision`.
        rc (dict): Recall dict returned by `recall`.
    Returns:
        dict: Containing precision and recall (macro and weighted).
    """
    return{
        'precision weighted': pr['weighted'],
        'precision macro': pr['macro'],
        'recall weighted': rc['weighted'],
        'recall macro': rc['macro']
    }