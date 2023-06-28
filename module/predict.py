import os
import cv2
import numpy as np
from tqdm import tqdm
from model_engine import YoloDetectPredict
from tracker_engine import DeepSortTracker

def predict_track(src_filepath: str,
                  dst_filepath: str,
                  model_path: str, 
                  conf_threshold=0.5,
                  iou_threshold=0.5,
                  max_age=60):
    """
    This function helps in predicting and tracking objects in a video.

    Parameters:
        src_filepath: A string directing towards the source video file.
        dst_filepath: A string directing towards the directory path where the 
                      processed video will be saved.
        model_path: A string directing towards the model file for performing 
                    inference on the video.
        conf_threshold: A float in the range (0, 1) for thresholding the confidence scores.
        iou_threshold: A float in the range (0, 1) for thresholding IoU while 
                       performing Non maximum suppression.
    """
    # Creating destination directory if not exist
    if not os.path.exists('/'.join(dst_filepath.split('/')[:-1])):
        os.makedirs('/'.join(dst_filepath.split('/')[:-1]))

    # Reading the video file and getting the metadata
    video_reader = cv2.VideoCapture(src_filepath)
    vid_wd = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_ht = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Creating a instance for writing a video file
    fourcc = 0x00000021 # H.264 Codec
    video_writer = cv2.VideoWriter(dst_filepath,
                                   fourcc, 
                                   video_reader.get(cv2.CAP_PROP_FPS),
                                   (vid_wd, vid_ht))
    
    # Initiating the model and tracking instance
    model = YoloDetectPredict(model_path=model_path,
                              conf_threshold=conf_threshold,
                              iou_threshold=iou_threshold)
    classes = model.get_meta_details()
    tracker = DeepSortTracker(max_age=max_age)

    # Generating colors for every class
    rng = np.random.default_rng(3) # Random number generator
    colors = rng.uniform(0, 255, size=(len(classes), 3))

    # scale and thickness for drawing objects
    thickness = 2
    text_thickness = 1
    text_scale = 0.5
    text_padding = 10
    text_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Reading all the frames in the video file
    progress_bar = iter(tqdm(range(total_frame_count)))
    while video_reader.isOpened():
        success, frame = video_reader.read()
        if not success:
            break
        
        # Performing prediction on the frame
        boxes, scores, class_ids = model(frame)
        
        # Accumulating box and score together
        det = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            det.append([x1, y1, x2, y2, score])
        
        # Updating tracker with the detected objects
        tracker.update(frame=frame, bbox_detection=det)

        # Getting the data from the tracker
        for track, class_id in zip(tracker.tracks, class_ids):
            bbox = track.bbox
            x1, y1, x2, y2 = bbox.astype(int)
            track_id = track.track_id
            color = colors[int(class_id)]

            # Drawing the detected object
            cv2.rectangle(img=frame,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=color,
                          thickness=thickness)
            
            # Writing a caption over every object
            caption = f'{classes[int(class_id)]}: #{track_id}'

            tw, th = cv2.getTextSize(text=caption,
                                     fontFace=font,
                                     fontScale=text_scale,
                                     thickness=text_thickness)[0]
            
            text_x, text_y = x1 + text_padding, y1 - text_padding

            text_box_x1 = x1
            text_box_y1 = y1 - 2 * text_padding - th
            text_box_x2 = x1 + 2 * text_padding + tw
            text_box_y2 = y1

            cv2.rectangle(img=frame,
                          pt1=(text_box_x1, text_box_y1),
                          pt2=(text_box_x2, text_box_y2),
                          color=color,
                          thickness=cv2.FILLED,
            )
            cv2.putText(img=frame,
                        text=caption,
                        org=(text_x, text_y),
                        fontFace=font,
                        fontScale=text_scale,
                        color=text_color,
                        thickness=text_thickness,
                        lineType=cv2.LINE_AA)
        next(progress_bar)
        video_writer.write(frame)
    video_writer.release()
    video_reader.release()
    print(f'[INFO] File "{src_filepath}" has been processed and saved in directory: "{dst_filepath}".')

if __name__ == '__main__':
    src_filepath = 'Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4'
    dst_dirpath = 'result/output.mp4'
    model_path = 'models/exp1_yolov8m/best.onnx'

    # Running inference and saving the result
    predict_track(src_filepath=src_filepath,
                  dst_dirpath=dst_dirpath,
                  model_path=model_path)
