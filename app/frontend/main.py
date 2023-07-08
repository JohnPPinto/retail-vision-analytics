import os
import cv2
import requests
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tempfile import TemporaryDirectory

# Setting page layout
st.set_page_config(page_title='Retail Vision Analytics',
                   page_icon='🏪',
                   layout='wide',
                   initial_sidebar_state='expanded')

# Page heading
st.title('Retail Vision Analytics')

##### Configuration - Page sidebar #####

# Predefining some values for the polygons
polygons = ''
thickness = 4
font_scale = 2
font_thickness = 4

with st.sidebar:
    st.header('Configuration:')
    
    # Upload or Choose video options
    video_choice = st.radio(label='Upload or Choose a Video', 
                            options=('Upload', 'Choose'), 
                            horizontal=True)

    if video_choice == 'Upload':
        # Display a file uploader widget
        video = st.file_uploader(label='Upload a video...', type=('mp4', 'mpg'))
    elif video_choice == 'Choose':
        # Display select box widget
        video = st.selectbox(label='Choose a video example...', 
                             options=('Example No. 1', 'Example No. 2'))

    # Display radio buttons for choosing the video process type
    process_choice = st.radio(label='Choose the video processing type...',
                              options=('OpenCV', 'FFMPEG'),
                              horizontal=True)
    
    # Display checkbox for polygon zone
    polygon_canvas = st.checkbox(label='Select for Polygon Zone')
    if polygon_canvas:
        thickness = st.slider(label='Border thickness:', min_value=1, max_value=10, value=4)
        font_scale = st.slider(label='Count text scale:', min_value=1, max_value=10, value=2)
        font_thickness = st.slider(label='Count text thickness:', min_value=1, max_value=10, value=4)

    # Display button for submitting the video to the backend
    submit_button = st.button('Submit', use_container_width=True)

##### Main Content - Page Body #####

# Tabs for displaying videos
tab1, tab2, tab3 = st.tabs(['Original', 'Canvas', 'Result'])

with tab2:
    if not polygon_canvas:
        st.info('Please upload a video and select the Polygon Zone box in the configuration.')

with tab1:
    # Display the input video file
    if video is not None:
        
        # Creating a temp directory and processing the video file 
        with TemporaryDirectory() as tmpdirname:
            # Reading the file
            if video_choice == 'Upload':
                contents = video.getvalue()
            elif video_choice == 'Choose':
                if video == 'Example No. 1':
                    with open('Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4', 'rb') as f:
                        contents = f.read()
                elif video == 'Example No. 2':
                    with open('OneLeaveShop1cor.mpg', 'rb') as f:
                        contents = f.read()
            
            with open(f'{tmpdirname}/tempfile.mp4', 'wb') as tempfile:
                # Writing the contents of the file in the temp file
                tempfile.write(contents)
            
                # Converting the video file and displaying it.
                os.system(f'ffmpeg -y -i {tmpdirname}/tempfile.mp4 -vcodec libx264 {tmpdirname}/output.mp4')
                st.video(f'{tmpdirname}/output.mp4')

                if polygon_canvas:
                    # Getting the first frame of the video and saving in temp dir
                    reader = cv2.VideoCapture(f'{tmpdirname}/tempfile.mp4')
                    vid_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                    _, img = reader.read()
                    cv2.imwrite(f'{tmpdirname}/output.png', img)
                    reader.release()

                    # Creating a canvas for drawing polygon
                    with tab2:
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=2,
                            background_image=Image.open(f'{tmpdirname}/output.png'),
                            height=vid_height,
                            width=vid_width,
                            drawing_mode='polygon',
                            point_display_radius=0,
                            key="canvas",
                        )

                        # Getting the json data from the canvas
                        if canvas_result.json_data is not None:
                            json_data = canvas_result.json_data
                            
                            data_length = []
                            coords = {}

                            # Collecting the shape of all the polygon
                            for i in json_data['objects']:
                                if i['type'] == 'path':
                                    data_length.append(len(i['path']))

                            # Collecting polygon coordinates if all the polygon have the same shape
                            for length in data_length:
                                if data_length[0] == length: # Verifying polygon shape
                                    for e, i in enumerate(json_data['objects']):
                                        coords[f'polygon: {e}'] = [] # list for collecting the coordinates
                                        if i['type'] == 'path':
                                            for j in i['path']:
                                                if len(j[1:]) > 0:
                                                    coords[f'polygon: {e}'].append(j[1:])
                                else:
                                    coords = {} # empty coords if shape of all the polygon are not same
                                    st.warning('All Polygons should have the same shape. Please clean the canvas and redraw the polygons.')
                                    break

                            # Appending all the polygon coordinates
                            polygons = []
                            if len(coords) > 0:
                                st.dataframe(coords) # Displaying the coordinates
                                for k, v in coords.items():
                                    polygons.append(v)
                            else:
                                polygons = ''
                            polygons = str(polygons) # Backend API accepts string

    else:
        st.info('Please upload a video!')

with tab3:
    # On submitting the configuration
    if submit_button:

        # Reading the video
        if video is not None:
            if video_choice == 'Upload':
                files = {'file': video.getvalue()}
            elif video_choice == 'Choose':
                if video == 'Example No. 1':
                    with open('Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4', 'rb') as f:
                        files = {'file': f.read()}
                elif video == 'Example No. 2':
                    with open('OneLeaveShop1cor.mpg', 'rb') as f:
                        files = {'file': f.read()}

            with st.spinner('Loading, this might take some time...'):
                # Sending a request to backend API
                params = {'process': process_choice, 
                          'polygons': polygons,
                          'thickness': thickness,
                          'font_scale': font_scale,
                          'font_thickness': font_thickness}
                res = requests.post(f'http://backend:8080/predict-and-track', files=files, params=params)
                video_path = res.json()

            # Displaying the result video file
            with open(video_path.get('filename'), 'rb') as video_file:
                video_bytes = video_file.read()
            if video_bytes:
                st.video(data=video_bytes)
        else:
            st.error("Please select a video before submitting!")
    else:
        st.info('Please submit the video!')