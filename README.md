July 14, 2023

# Local Installation

* Clone the repository
```
https://github.com/JohnPPinto/retail-vision-analytics.git
```
* Change the working directory
```
cd retail-vision-analytics/app
```
* Run the docker compose file (Docker and Docker compose should be installed)
```
docker compose up -d --build
```
* Or you can pull the docker images
```
docker compose pull
docker compose up -d
```
* Navigate to the frontend and backend
```
http://localhost:8501
http://localhost:8080/docs
```

# **Project Retail Vision Analytics**

This project is a demonstration of my ability to work in the field of Deep Learning, showcase my skills to learn quickly, and develop and deploy end-to-end Artificial Intelligence technologies. Currently, this project is still in work in progress phase, project has completed the first stage and will be moving toward the continuous training (CT) stage, where I will be going back to the data pipeline and iterating improvements to all of the pipelines.

## **Project Objective**

Many retail companies and stores have cameras, these cameras have only a single purpose which is to provide security and detection once the issue has occurred. However, the data that the retailers have is much more valuable than only providing security footage. This data can provide keen insight that can give them an edge in their competition and provide the ultimate customer satisfaction.

So the question that comes out is, why are retailers not using video data for analyzing their customers and inventories? this is because working on video data manually can be a tedious task and over time it can be challenging and the cost of the project can increase.

Retailer's main objective is to understand their customers and the best way to do that is by using the video data they have collected, working on the video data a few years back would have been difficult but now using neural networks it is possible to do the same task in less time and save resources at the same time.

## **Data Pipeline**

The data pipeline is the most important stage in an ML project. The data pipeline dictates how the ML project will behave and what sort of quality prediction the model will generate.

For this project, initially, I searched for a publicly available dataset that shows a retail store but there were no data available or generated. Then I looked for similar environment data but not limited to stores, and I was able to find a public dataset. The primary video dataset I have used for this project is the [CAVIAR Dataset](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/), This publicly available dataset contains videos collected from a shopping mall hallway and mall stores, two fixed CCTV cameras are used which give the same features that a retail store will have.

Once I has the data, I generated the annotation using the [CVAT](https://www.cvat.ai/) application, framing all the objects using CVAT is easy and fast for the video dataset. There are some limitations to CVAT but it is still possible to generate good quality annotation for this project. CVAT also provides different types of formats to export annotation this makes the preprocessing steps much more easier and faster. Along with the annotation, I even reviewed and assessed the videos manually.

## **Model Pipeline**

The Model pipeline is all about selecting the model, training the model, and evaluating the model's performance. Similar to the data pipeline this is a crucial stage where the performance of the model is the core of the project. A bad-performing model is of no use to anyone and going further with a bad model will degrade the overall project objective and waste the resource allocated for it.

Starting with the model pipeline, I have used the [YOLOv8](https://github.com/ultralytics/ultralytics) model for generating object detection of the person in the video dataset. At the time of building this project, YOLOv8 is one of the state-of-the-art models for computer vision tasks. While training the model for object detection I was able to achieve the following result: **Precision: 0.991, Recall: 0.991, mAP50: 0.995, and mAP50-95: 0.976**. After evaluating the model, I exported the model in ONNX format for further development. You can read about ONNX on their website over [here](https://onnx.ai/).

Along with object detection, I also needed a way to track the object in the whole stream of video, so to achieve this I used DeepSORT, DeepSORT is capable to track any object by placing an ID on them, but DeepSORT have some limitation also like too many switching of ID's and poor tracking on occlusions in a group of people, this results in quality drop for object tracking.

## **Application Prototype**

The frontend of the project is built using [Streamlit](https://streamlit.io/), for fast prototyping Streamlit is the best way to handle the frontend and to communicate with the backend I have used FastAPI, to generate a prediction on request by a user. The user can access the whole application on the web app, upload their videos and draw the polygon area for tracking and counting the customers and visualize the result in a video format, all of this in a few clicks. I have also added Dockerfiles and docker-compose files for the project, this will make the whole process to deploy on the cloud simple and fast.

## **Conclusion**

This project showcases the possibility and reliability of today's computer vision algorithms and methods to automate the process of analyzing videos, these techniques can be even used wherever the camera is present whether the data is old or real-time, and can improve the same task that was manually performed.

Currently, there are a few limitations in this project, starting with the data pipeline, while working on this project I was able to understand that the objects are the only person that is needed but the environmental factor is also important, more data is required with people carrying different goods, this will help the model in generalizing the data and annotation needs to contain the objects that the person holds with them along. For the model pipeline, improvement needs to be done in the tracking algorithm, when in a group the tracking quality should not drop. There is even a chance that the model won't be able to differentiate if there is a human-like object present in the video, which can again affect the overall result.

Now for the inference and deployment pipeline, this project has the biggest limitation, all the processing is dependent on the hardware, the better the hardware the faster the calculations and the quicker the result is generated. While processing the video the model reads every frame and applies the result to the frames which makes it time-consuming, there are some ways to improve speed but at the cost of video quality degradation, which can be a sensitive factor to consider.

To date, the project is not up to the mark, currently it even needs a way to display all the analytics in a dashboard, and additional features like heatmaps, object movement and flow tracking, object time-interval, and many more. But with this, we can show that retailers now have a way to improve and optimize their store layouts and management decisions which will ultimately improve the customer experience.
