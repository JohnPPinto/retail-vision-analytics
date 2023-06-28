import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DSTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

class DeepSortTracker:
    """
    A Deep Sort tracker for tracking multiple objects.
    
    Parameters:
        max_age: A int for stating the tracking age, tracker to track the object for certain number of frames.
    """
    tracker = None
    encoder = None
    tracks = None
    
    def __init__(self, max_age=60):
        self.max_age = max_age
        max_cosine_distance = 0.4
        nn_budget = None
        
        # Initializing the tracker
        model_filename = 'models/deep_sort/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename=model_filename,
                                               batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(metric='cosine', 
                                                            matching_threshold=max_cosine_distance, 
                                                            budget=nn_budget)
        self.tracker = DSTracker(metric=metric, max_age=self.max_age)
    
    def update(self, frame, bbox_detection):
        # Getting the bounding box data
        bboxes = np.asarray([d[:-1] for d in bbox_detection])
        
        # Getting the width and height of the bounding box (bbox_detection needs to be in Pascal VOC format)
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        
        # Getting the score of the bbox
        scores = [d[-1] for d in bbox_detection]
        
        # Generating deep sort feature for every object in the bbox
        features = self.encoder(frame, bboxes)
        detections = []
        for bbox_id, bbox in enumerate(bboxes):
            detections.append(Detection(tlwh=bbox, 
                                        confidence=scores[bbox_id], 
                                        feature=features[bbox_id]))
            
        # Performing deep sort detection and updating it for every step or frame
        self.tracker.predict()
        self.tracker.update(detections=detections)
        self.update_tracks()
        
    def update_tracks(self):
        # Updating tracker data
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # Converting bbox data from COCO to Pascal format
            bbox = track.to_tlbr()
            
            # Getting the tracker ID
            id = track.track_id
            
            # Appending the tracker data 
            tracks.append(Track(id, bbox))
        self.tracks = tracks
        
class Track:
    track_id = None
    bbox = None
    
    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
