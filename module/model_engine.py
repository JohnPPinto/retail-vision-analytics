import math
import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from utils import rescale_boxes, bbox_yolo_to_pascal, clip_bbox, compute_nms, draw_detections

class YoloDetectPredict:
    """
    This class helps in loading the model, predicting objects in the image and providing bounding
    box coords for all the detected objects in the image.
    Parameters:
        model_path: A string to the path directing towards the model location.
        conf_threshold: A float in the range (0, 1) for thresholding the confidence scores.
        iou_threshold: A float in the range (0, 1) for thresholding IoU while Non maximum supression.
    """
    def __init__(self, model_path, conf_threshold = 0.7, iou_threshold = 0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initializing the model
        self.initialize_model(model_path)
        
    def initialize_model(self, model_path):
        # Initializing onnx model instance
        EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, 
                                                        providers = EP_LIST)
        # Get meta data from the model
        self.get_meta_details()
        self.get_input_details()
        self.get_output_details()
        
    def get_meta_details(self):
        # Getting the model meta data.
        model_meta = self.ort_session.get_modelmeta()
        self.class_dict = eval(model_meta.custom_metadata_map['names'])
        self.class_list = list(self.class_dict.values())
        return self.class_list
    
    def get_input_details(self):
        # Getting the input data
        model_inputs = self.ort_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        # Getting the output data
        model_outputs = self.ort_session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
    def __call__(self, image):
        # Performing prediction on the image
        return self.detect_objects(image)
    
    def detect_objects(self, image):
        # Prepare the image array as a input tensor.
        input_tensor, self.input_img_resized = self.prepare_input(image)
        
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        
        # Extract prediction data
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        
        return self.boxes, self.scores, self.class_ids
    
    def prepare_input(self, image):
        # Getting image info
        self.image_height, self.image_width = image.shape[:2]
        
        # Resize input image to input size
        input_img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Preprocessing the input image
        input_img = input_img_resized / 255.0 # Normalizing
        input_img = input_img.transpose(2, 0, 1) # Converting the image into CHW format
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32) # Batching
        
        return input_tensor, input_img_resized
    
    def inference(self, input_tensor):
        # Predicting using the Yolo onnx model
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        return outputs
    
    def process_output(self, output):
        # Extracting predictions from box outputs
        predictions = np.squeeze(output).T # Transposing the data into (prediction, features)
        
        # Filter out on confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        # Validating for no prediction
        if len(scores) == 0:
            return [], [], []
        
        # Getting class with the highest confidense score
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Getting the bounding box for all the objects
        boxes = self.extract_boxes(predictions)
        
        # Apply Non Maximum Supression(NMS) to suppress overlapping box
        indices = compute_nms(boxes=boxes, 
                              scores=scores, 
                              iou_threshold=self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]
    
    def extract_boxes(self, predictions):
        # Extract box from predictions
        boxes = predictions[:, :4]
        
        # Scale boxes to original image dimension
        boxes = rescale_boxes(boxes=boxes, 
                              input_shape=(self.input_height, self.input_width), 
                              output_shape=(self.image_height, self.image_width))
        
        # Convert the boxes to pascal voc format
        boxes = bbox_yolo_to_pascal(boxes=boxes)
        
        # Clipping the boxes range to a image limit
        boxes = clip_bbox(boxes=boxes, 
                          height=self.image_height, 
                          width=self.image_width)
        
        return boxes
    
    def draw_bbox(self, image):
        # Drawing the predicted bounding box.
        return draw_detections(image=image,
                               boxes=self.boxes,
                               scores=self.scores,
                               class_ids=self.class_ids,
                               class_list=self.class_list)
