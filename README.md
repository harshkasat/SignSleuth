# Hand Sign Detection

* Capturing Images or Videos: The first step is to capture images or videos of hand signs using a camera or webcam.

* Preprocessing the Images: The captured images or videos are then preprocessed using OpenCV to remove noise and enhance the quality of the images.

* Hand Detection: OpenCV is used to detect the hands in the images or videos using algorithms such as CVzone library.

* Hand Tracking: Once the hands are detected, OpenCV is used to track the movement of the hands in the images or videos.

* Image Segmentation: The images are then segmented to separate the hands from the background using OpenCV.

* Feature Extraction: Features such as hand shape, finger position, and palm orientation are extracted from the segmented images using OpenCV.

* Machine Learning Model: A machine learning model is built using TensorFlow and Keras to classify the hand signs based on the extracted features.

* Training the Model: The machine learning model is trained using a dataset of hand signs.

* Testing the Model: The trained model is tested on a separate dataset of hand signs to evaluate its accuracy.

## Description

* The importance of dataset creation: Creating a high-quality dataset of hand signs is crucial for the accurate training of the machine learning model. The dataset must be diverse, representative, and balanced to ensure that the model is trained on a variety of hand signs.

* The power of transfer learning: I learned how transfer learning can be used to leverage pre-trained models to classify hand signs accurately. This approach saves time and computational resources, allowing for faster model development.

* The importance of hyperparameter tuning: Fine-tuning the hyperparameters of the machine learning model is essential to achieve optimal performance. I learned how to adjust the learning rate, batch size, and other hyperparameters to improve the model's accuracy.

* The effectiveness of OpenCV for hand detection: OpenCV provides powerful algorithms such as Haar Cascade and YOLO for hand detection, which can accurately detect and track hands in images and videos.

* The importance of model evaluation: Evaluating the model's accuracy on a separate dataset is crucial to ensure that the model is performing well and can be used in real-world applications.

* Overall, creating this code sample was an inspirational experience as it showed me the power of machine learning and computer vision in solving real-world problems. The project reinforced my confidence in my ability to develop complex machine learning models and use them to make a positive impact.

|    Dir name   |    Working                |
| ------------- | ------------------------- |
| DATA          | Dataset(hand-Sign)        |
| Model         | Keras Model & labels.txt  |
| handdetection | Venv                      |
| main.py       | To create Dataset         |
| test.py       | To run Dataset            |
| requirements.txt       | Requirement for program   |

## Getting Started

### Dependencies

* Python 3.8: The code requires Python 3.8 or higher to run.

* OpenCV: OpenCV is an open-source computer vision library used to process and analyze images and videos. It can be installed using the command "pip install opencv-python".

* TensorFlow: TensorFlow is an open-source machine learning framework developed by Google. It can be installed using the command "pip install tensorflow".

* Keras: Keras is a high-level neural networks API, written in Python, that runs on top of TensorFlow. It can be installed using the command "pip install keras".

* NumPy: NumPy is a Python library used for working with arrays. It can be installed using the command "pip install numpy".

* Matplotlib: Matplotlib is a plotting library for Python. It can be installed using the command "pip install matplotlib".

* requirements.txt file: This file contains all the dependencies needed to run the code. It can be installed using the command "pip install -r requirements.txt".

### Installing
* clone repository :- type in cmd terminal
```
git clone https://github.com/harshkasat/SignSleuth
cd SignSleuth
```


### Run Program

* activate venv
```
python -m venv handdetection
.\handdetection\Scripts\activate
```
* install requirements.txt 
```
python install requirements.txt
```
* Run Program
```bash
test.py
```

|    IDe            |    Run Code    |
| ----------------- | -------------- |
| Pycharm           | shift + f10    |
| IDLE              | f5             |
| jupyter notebook  | shift + enter  |

## Authors

Contributors names and contact info
* ex. [@HarshKasat](https://twitter.com/harsh__kasat)
