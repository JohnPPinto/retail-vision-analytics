import os
import uuid
import uvicorn
import json
import numpy as np
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from tempfile import NamedTemporaryFile

from predict import PredictTrack

app = FastAPI()

@app.get("/")
async def read_root():
    return {'message': 'Welcome to the Retail Vision API'}

@app.post('/predict-and-track')
async def predict_track_video(process: str,
                              polygons: str = '',
                              thickness: int = 4,
                              font_scale: int = 2,
                              font_thickness: int = 4,
                              position: str = 'bottom',
                              file: UploadFile = File(...)):
    """
    This API will take a video and run a inference to predict person objects
    and track all the objects providing a track id to every object. The processed
    video will be then be saved and the location will be returned.

    Parameters:
        process: A string to select the processing type for the video['OpenCV', 'FFMPEG'].
        polygons: A string containing the array of the polygon, 
                  shape of the array [number of polygon, number of points, xy coordinates].
        file: A UploadFile argument for uploading any local videos files.
    """
    model_path = 'models/exp1_yolov8m/best.onnx'
    result_filepath = f'/app/storage/{str(uuid.uuid4())}.mp4'

    # Creating a temporary file instance for the input file
    temp = NamedTemporaryFile(delete=False)
    
    # Reading the file
    contents = file.file.read()
    with temp as f:
        # Writing the contents of the file in the temp file
        f.write(contents);
    # Closing the input file
    file.file.close()
    
    # Creating inference instance
    inference = PredictTrack(src=temp.name,
                             dst=result_filepath,
                             model_path=model_path,
                             conf_threshold=0.5,
                             iou_threshold=0.5,
                             max_age=60)

    # Performing inference and generating result
    if len(polygons) > 0:
        polygons = np.asarray(json.loads(polygons))
    else:
        polygons = None
        
    if process == 'OpenCV':
        time_result = inference.process_video_opencv(polygons=polygons,
                                                     thickness=thickness,
                                                     font_scale=font_scale,
                                                     font_thickness=font_thickness,
                                                     position=position)
    elif process == 'FFMPEG':
        time_result = inference.process_video_ffmpeg(polygons=polygons,
                                                     thickness=thickness,
                                                     font_scale=font_scale,
                                                     font_thickness=font_thickness,
                                                     position=position)
    else:
        print('[ERROR] Please enter correct process, either "OpenCV" or "FFMPEG".\n')

    # Removing the temporary file
    os.remove(temp.name)

    return {'filename': result_filepath,
            'inference_time_result': time_result}

if __name__=='__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8080)