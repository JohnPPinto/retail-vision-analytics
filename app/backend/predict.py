import os
import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm
from time import perf_counter
from model_engine import YoloDetectPredict
from tracker_engine import DeepSortTracker
from utils import draw_box_label, draw_polyzone


class PredictTrack:
    """
    A class to helps in predicting and tracking objects in a video.

    Attributes:
        src: A string directing towards the source video file.
        dst: A string directing towards the directory path where the 
                      processed video will be saved.
        model_path: A string directing towards the model file for performing 
                    inference on the video.
        conf_threshold: A float in the range (0, 1) for thresholding the 
                        confidence scores.
        iou_threshold: A float in the range (0, 1) for thresholding IoU while 
                       performing Non maximum suppression.
        max_age: A int for stating the tracking age, tracker to track the 
                 object for certain number of frames.
    """
    def __init__(self,
                 src: str,
                 dst: str,
                 model_path: str,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 max_age: int = 60):
        self.src = src
        self.dst = dst
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        # Creating destination directory if not exist
        if not os.path.exists('/'.join(self.dst.split('/')[:-1])):
            os.makedirs('/'.join(self.dst.split('/')[:-1]))
    
    def initialize_model(self):
        # Initiating the model and tracking instance
        self.model = YoloDetectPredict(model_path=self.model_path,
                                       conf_threshold=self.conf_threshold,
                                       iou_threshold=self.iou_threshold)
        self.classes = self.model.get_meta_details()
        self.tracker = DeepSortTracker(max_age=self.max_age)
    
    def process_video_opencv(self, 
                             polygons = None,
                             thickness = 4,
                             font_scale = 2,
                             font_thickness = 4,
                             position = 'bottom'):
        
        self.initialize_model()
        reader, writer, total_frames = self.video_io_opencv()
        self.inference_time_list = self.generate_video_opencv(reader, 
                                                              writer, 
                                                              total_frames, 
                                                              polygons,
                                                              thickness,
                                                              font_scale,
                                                              font_thickness,
                                                              position)
        return self.inference_time_list
    
    def process_video_ffmpeg(self,
                             polygons = None,
                             thickness = 4,
                             font_scale = 2,
                             font_thickness = 4,
                             position = 'bottom'):
        
        self.initialize_model()
        video_array, writer = self.video_io_ffmpeg()
        self.inference_time_list = self.generate_video_ffmpeg(video_array, 
                                                              writer,
                                                              polygons,
                                                              thickness,
                                                              font_scale,
                                                              font_thickness,
                                                              position)
        return self.inference_time_list

    def video_io_opencv(self):
        # Reading the video file and getting the metadata
        video_reader = cv2.VideoCapture(self.src)
        self.vid_wd = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_ht = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)

        # Creating a instance for writing a video file
        fourcc = cv2.VideoWriter_fourcc(*'vp09') # vp80 Codec - WebM
        video_writer = cv2.VideoWriter(self.dst,
                                       fourcc, 
                                       fps,
                                       (self.vid_wd, self.vid_ht))
        return video_reader, video_writer, total_frames
    
    def generate_video_opencv(self, 
                              reader, 
                              writer, 
                              total_frames, 
                              polygons = None, 
                              thickness = 4,
                              font_scale = 2,
                              font_thickness = 4,
                              position = 'bottom'):
    
        # Generating colors for every class
        rng = np.random.default_rng(0) # Random number generator
        colors = rng.uniform(0, 255, size=(len(self.classes), 3))

        # Generating colors for Polyzone
        if polygons is not None:
            rng = np.random.default_rng(1234) # Random number generator
            poly_colors = rng.uniform(0, 255, size=(len(polygons), 3))


        # Tracking the inference on each frame
        progress_bar = iter(tqdm(range(total_frames)))
        total_inference_time = []
        start_time = perf_counter()

        # Reading all the frames in the video file
        while reader.isOpened():
            success, frame = reader.read()
            if not success:
                break
            
            # Performing prediction on the frame
            boxes, scores, class_ids = self.model(frame)

            # Accumulating box and score together
            det = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.astype(int)
                det.append([x1, y1, x2, y2, score])

            # Updating tracker with the detected objects
            self.tracker.update(frame=frame, bbox_detection=det)

            # Getting the data from the tracker
            for track, class_id in zip(self.tracker.tracks, class_ids):
                bbox = track.bbox
                x1, y1, x2, y2 = bbox.astype(int)
                track_id = track.track_id
                color = colors[int(class_id)]

                # A caption for every object
                caption = f'{self.classes[int(class_id)]}: #{track_id}'

                # Drawing box and label for every object
                frame = draw_box_label(image=frame,
                                       caption=caption,
                                       x1=x1,
                                       y1=y1,
                                       x2=x2,
                                       y2=y2,
                                       color=color)

            # Drawing polygon zone 
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    frame = draw_polyzone(image=frame,
                                          polygon=polygon,
                                          bbox=boxes,
                                          image_wh=(self.vid_wd + 1, self.vid_ht + 1),
                                          color=poly_colors[i],
                                          thickness=thickness,
                                          font_scale=font_scale,
                                          font_thickness=font_thickness,
                                          position=position)                

            # Updating the new frame in the writer video file
            writer.write(frame)

            # Updating time tracker
            stop_time = perf_counter()
            total_inference_time.append(stop_time - start_time)
            next(progress_bar)
            start_time = perf_counter()

        writer.release()
        reader.release()
        print(f'[INFO] File "{self.src}" has been processed and saved in directory: "{self.dst}".')
        return total_inference_time
    
    def video_io_ffmpeg(self):
        # Creating a video instance
        probe = ffmpeg.probe(self.src)

        # Getting meta data of the video
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        fps = int(video_info['r_frame_rate'].split('/')[0])

        # Creating a ffmpeg video read instance
        read_buffer, _ = (
            ffmpeg
            .input(self.src)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )

        # Converting the buffer into array[n,h,w,c]
        video_array = (
            np
            .frombuffer(read_buffer, np.uint8)
            .reshape([-1, self.height, self.width, 3])
        )

        # Creating a ffmpeg video write instance
        write_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.width}x{self.height}')
            .output(self.dst, pix_fmt='yuv420p', vcodec='libx264', r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return video_array, write_process
    
    def generate_video_ffmpeg(self, 
                              video_array, 
                              writer,
                              polygons = None, 
                              thickness = 4,
                              font_scale = 2,
                              font_thickness = 4,
                              position = 'bottom'):
        
        # Generating colors for every class
        rng = np.random.default_rng(0) # Random number generator
        colors = rng.uniform(0, 255, size=(len(self.classes), 3))

        # Generating colors for Polyzone
        if polygons is not None:
            rng = np.random.default_rng(1234) # Random number generator
            poly_colors = rng.uniform(0, 255, size=(len(polygons), 3))

        # Tracking the inference on each frame
        total_inference_time = []
        start_time = perf_counter()

        # Reading all the frames in the video array
        for frame in tqdm(video_array):
            # Performing prediction on the frame
            boxes, scores, class_ids = self.model(frame)

            # Accumulating box and score together
            det = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.astype(int)
                det.append([x1, y1, x2, y2, score])

            # Updating tracker with the detected objects
            self.tracker.update(frame=frame, bbox_detection=det)

            # Getting the data from the tracker
            for track, class_id in zip(self.tracker.tracks, class_ids):
                bbox = track.bbox
                x1, y1, x2, y2 = bbox.astype(int)
                track_id = track.track_id
                color = colors[int(class_id)]

                # A caption for every object
                caption = f'{self.classes[int(class_id)]}: #{track_id}'

                # Drawing box and label for every object
                frame = draw_box_label(image=frame,
                                       caption=caption,
                                       x1=x1,
                                       y1=y1,
                                       x2=x2,
                                       y2=y2,
                                       color=color)

            # Drawing polygon zone 
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    frame = draw_polyzone(image=frame,
                                          polygon=polygon,
                                          bbox=boxes,
                                          image_wh=(self.width + 1, self.height + 1),
                                          color=poly_colors[i],
                                          thickness=thickness,
                                          font_scale=font_scale,
                                          font_thickness=font_thickness,
                                          position=position)                

            # Updating the new frame in the writer video file
            writer.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )

            # Updating time tracker
            stop_time = perf_counter()
            total_inference_time.append(stop_time - start_time)
            start_time = perf_counter()

        writer.stdin.close()
        writer.wait()
        print(f'[INFO] File "{self.src}" has been processed and saved in directory: "{self.dst}".')
        return total_inference_time

if __name__ == '__main__':
    src = 'Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4'
    dst = 'result/output.mp4'
    model_path = 'models/exp1_yolov8m/best.onnx'

    # polygon coordinates
    polygons = np.asarray([[[130, 82],[522, 74],[518, 554],[98, 542]],
                          [[758, 218],[1218, 218],[1034, 670],[722, 394]]])

    # Running inference and saving the result
    time_result = PredictTrack(src=src,
                               dst=dst,
                               model_path=model_path).process_video_ffmpeg(polygons=polygons)