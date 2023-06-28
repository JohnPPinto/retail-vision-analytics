import numpy as np
import cv2

# Rescale any bounding box
def rescale_boxes(boxes, input_shape, output_shape):
    """
    This functions helps in re-scaling bounding box from one object to another.
    
    Parameters:
        boxes: An array containing the values of the bounding box.
        input_shape: A tuple or list containing values of the original object shape. E.g. (height, width)
        output_shape: A tuple or list containing values of the output object shape. E.g. (height, width)
    
    Returns:
        boxes: An array containing the values of the rescale boxes.
    """
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([output_shape[1], output_shape[0], output_shape[1], output_shape[0]])
    return boxes

# Convert bounding box from YOLO format (x_c, y_c, w, h) into Pascal VOC format (x1, y1, x2, y2)
def bbox_yolo_to_pascal(boxes):
    """
    This function helps in converting the bounding box format from YOLO to Pascal VOC.
    
    Parameters:
        boxes: An array containing the values of the bounding box in YOLO format.
    
    Returns:
        boxes_cp: An array containing the values of the bounding box in Pascal VOC format.
    """
    boxes_cp = boxes.copy()
    boxes_cp[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes_cp[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes_cp[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    boxes_cp[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return boxes_cp

# Clipping the bounding box values
def clip_bbox(boxes, height, width):
    """
    This function helps in clipping the values of the bounding box.
    
    Parameters:
        boxes: An array containing the values of the bounding box. 
        height: An int value of the height of a Image or Frame.
        width: An int value of the width of a Image or Frame.
    
    Return:
        clip_boxes: An array containing the clipped values of the bounding box.
    """
    clip_boxes = boxes.copy()
    clip_boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
    clip_boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
    clip_boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
    clip_boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
    return clip_boxes

# Computing the Intersection over Union of the bounding box.
def compute_iou(box, boxes):
    """
    This function helps in calculating the intersection over union of the bounding boxes.
    This function best works with prediction result, where one predicted box is computed with 
    multiple different predicted boxes.
    
    Parameters:
        box: An array containing values of a bounding box.
        boxes: An array containing values of multiple different bounding box.
    
    Returns:
        iou: An array containing iou values in between range (0, 1) for all the boxes array.
    """
    # Getting the intersection box
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    
    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    
    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area
    return iou

# Computing Non Maximum Suppression on all the bounding box
def compute_nms(boxes, scores, iou_threshold):
    """
    This function helps in computing the Non Maximum Suppression on the 
    predicted bounding boxes.
    
    Parameters:
        boxes: An array containing the values of the bounding boxes.
        scores: An array containing the values of the confidence scores
                for each bounding box.
        iou_threshold: A float value to suppress the bounding box.
                       Value should be within the range (0, 1).
    
    Returns: 
        Keep_boxes: A list containing the index for the boxes and scores 
                    array after computing Non Maximum Suppression.
    """
    # Getting the list of indices of sorted scores - descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    # Looping through the indices and computing nms
    keep_boxes = []
    while sorted_indices.size > 0:
        # Picking the box with best score
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        # Compute IoU of the picked box with rest of the boxes
        ious = compute_iou(box=boxes[box_id, :], boxes=boxes[sorted_indices[1:], :])
        
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        
        # Keeping only the indices that fit within the threshold
        sorted_indices = sorted_indices[keep_indices + 1]
        
    return keep_boxes

def draw_detections(image, boxes, scores, class_ids, class_list):
    """
    This function helps in drawing the predicted detection bounding box on the image.
    
    Parameters:
        image: An array containing the values of the base image in RGB format.
        boxes: An array containing the values of the predicted bounding box in Pascal Voc format.
        scores: An array containing the values of the confidence score for each predicted bounding box.
        class_ids: An array containing the values of the predicted classes indices. 
        class_list: A list containing all the class names in proper order.

    Returns:
        image: An array containing the values for the image with the predicted bounding box drawn for every object.
    """
    image_height, image_width = image.shape[:2]
    size = min([image_height, image_width]) * 0.001 # Dynamic fontscale
    text_thickness = int(min([image_height, image_width]) * 0.001) # Dynamic thickness
    
    # Generating colors for every class
    rng = np.random.default_rng(3) # Random number generator
    colors = rng.uniform(0, 255, size=(len(class_list), 3))
    
    # Draw predicted bounding box and labels on the image
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        
        # Drawing rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Getting the box coords of the label text
        label = class_list[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, 
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                      fontScale=size, 
                                      thickness=text_thickness)
        th = int(th * 1.2)
        
        # Drawing rectangle for the text
        cv2.rectangle(image, 
                      (x1, y1), 
                      (x1 + tw, y1 - th if y1 - 10 > 0 else y1 + 10 + th), 
                      color, 
                      -1)
        
        # Adding the label text
        cv2.putText(image, 
                    caption, 
                    (x1, y1 if y1 - 10 > 0 else y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    size, 
                    (255, 255, 255), 
                    text_thickness)
    return image
