import requests
import streamlit as st

# Setting page layout
st.set_page_config(page_title='Retail Vision Analytics',
                   page_icon='🏪',
                   layout='wide',
                   initial_sidebar_state='collapsed')

# Page heading
st.title('Retail Vision Analytics')

# Page columns
col1, col2 = st.columns(2)

with col1:
    # Display a file uploader widget
    video = st.file_uploader(label='Choose a video...', type=('mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'mpg'))

    # Display radio buttons for choosing the video process type
    process_choice = st.radio(label='Choose the video processing type...',
                              options=('OpenCV', 'FFMPEG'),
                              horizontal=True)
    
    # Display button for submitting the video to the backend
    submit_button = st.button('Submit', use_container_width=True)

with col2:

    # Tabs for displaying videos
    tab1, tab2 = st.tabs(['Original', 'Processed'])

    with tab1:
        # Display the input video file
        if video is not None:
            # Reading the video files
            video_bytes_data = video.getvalue()
            st.video(data=video_bytes_data)
        else:
            st.info('Please upload a video!')

    with tab2:
        # Display a button to process the video
        if submit_button:

            # Reading the video
            if video is not None:
                files = {'file': video.getvalue()}

                with st.spinner('Loading, this might take some time...'):
                    # Sending a request to backend API
                    params = {'process': process_choice}
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