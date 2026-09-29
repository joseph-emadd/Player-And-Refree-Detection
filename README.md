# ⚽ Football Player Detection & Tracking

A computer vision project for detecting and tracking football players, goalkeepers, and referees in match videos using **YOLO11** and **ByteTrack**.

## Project Overview

This project builds an object detection and tracking pipeline for football match footage.

The model is trained to detect three classes:

* **GoalKeeper**
* **Player**
* **Referee**

After training, the best-performing YOLO model is used to track detected objects in a football match video using the **ByteTrack** tracking algorithm.

## 🛠️ Technologies Used

* Python
* Google Colab
* PyTorch
* Ultralytics YOLO11
* ByteTrack
* OpenCV
* Google Drive

## 📂 Dataset

The dataset uses the YOLO annotation format.

Each annotation contains:

```text
class_id x_center y_center width height
```

The three classes are:

| Class ID | Class      |
| -------- | ---------- |
| 0        | GoalKeeper |
| 1        | Player     |
| 2        | Referee    |

Before training, the annotations are validated to check:

* Correct annotation format
* Valid class IDs
* Numeric bounding-box values
* Bounding-box values within the `[0, 1]` range

## 🔄 Data Preparation

The dataset is divided into:

* **80% Training**
* **20% Validation**

A fixed random seed is used to make the split reproducible.

A `data.yaml` configuration file is then created for YOLO training.

## 🤖 Model Training

The project uses the lightweight **YOLO11n** model as the base model.

Training configuration:

* **Model:** YOLO11n
* **Epochs:** 100
* **Image Size:** 640 × 640
* **Batch Size:** 16
* **Patience:** 20 epochs
* **GPU:** CUDA device

The best trained model is saved as:

```text
training/football_yolo/weights/best.pt
```

## 🎥 Object Tracking

After training, the best model is used to detect and track objects in a football match video.

The project uses:

**ByteTrack**

with:

* Confidence threshold: `0.25`
* IoU threshold: `0.5`
* Persistent tracking enabled

The tracking output is saved to the `tracking` directory.

## 📊 Pipeline

```text
Football Dataset
       ↓
Label Validation
       ↓
Train / Validation Split
       ↓
YOLO11n Training
       ↓
Best Model (best.pt)
       ↓
Football Match Video
       ↓
Object Detection
       ↓
ByteTrack
       ↓
Tracked Video
```

## 🚀 How to Run

### 1. Open the Notebook

Open `Player_Detection.ipynb` in Google Colab.

### 2. Connect Google Drive

The notebook uses Google Drive to store the dataset, trained model, videos, and results.

Expected project structure:

```text
player detection/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── videos/
│   └── match.mp4
│
├── training/
│   └── football_yolo/
│       └── weights/
│           └── best.pt
│
└── tracking/
```

### 3. Install Dependencies

The notebook installs Ultralytics automatically:

```python
!pip install -q ultralytics
```

### 4. Train the Model

Run the training cells to train YOLO11n on the football dataset.

### 5. Track a Match Video

Place your football video inside the `videos` folder and update the video path if necessary.

The trained model will then be used with ByteTrack to generate the tracked video.

## 📁 Files

| File                     | Description                             |
| ------------------------ | --------------------------------------- |
| `Player_Detection.ipynb` | Complete training and tracking pipeline |
| `best.pt`                | Best trained YOLO model                 |
| `data.yaml`              | Dataset configuration                   |
| `match.mp4`              | Input football match video              |
| `match.avi`              | Output tracked video                    |

## 🎯 Future Improvements

Possible improvements include:

* Evaluate the model using mAP, Precision, and Recall
* Add more football datasets
* Improve detection of players under occlusion
* Track individual players more consistently
* Add player statistics such as movement and distance
* Detect teams based on jersey colors
* Add real-time detection from live video

⭐ If you find this project useful, feel free to star the repository.
