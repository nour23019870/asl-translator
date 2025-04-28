# ASL Translator

A real-time American Sign Language (ASL) translator that uses computer vision and deep learning to recognize and interpret hand gestures.

## 📋 Overview

This project utilizes MediaPipe for hand landmark detection and a custom CNN model to classify hand gestures into American Sign Language (ASL) letters and numbers. The system can detect and translate:

- Numbers (0-9)
- Letters (A-Z)

The application processes webcam input in real-time, extracts hand landmarks, and provides translation with confidence scores.

## ✨ Features

- Real-time hand tracking and landmark detection
- ASL gesture recognition for letters and numbers
- Prediction smoothing for stable outputs
- Visual feedback with confidence measurement
- Interactive user interface

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- NumPy
- pandas (for training)
- scikit-learn (for training)

## 🚀 Installation

1. Clone this repository
```bash
git clone https://github.com/nour23019870/asl-translator.git
cd asl-translator
```

2. Install dependencies
```bash
pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn
```

## 💻 Usage

### Real-time ASL Translation

Run the main application to start translating ASL gestures in real-time:

```bash
python main.py
```

This will open your webcam and start detecting and translating hand gestures.

### Hand Tracking Demo

To simply see hand landmark detection without translation:

```bash
python hand_tracking_cam.py
```

### Training Your Own Model

1. Prepare your dataset in the required format (images organized in class folders)
2. Run the dataset creation script:

```bash
python create_landmark_dataset.py
```

3. Train the classifier:

```bash
python train_asl_classifier.py
```

## 🧠 How It Works

1. **Hand Detection**: MediaPipe detects hand presence in the video frame
2. **Landmark Extraction**: 21 key points are identified on the hand
3. **Skeleton Visualization**: The landmarks are connected to form a hand skeleton
4. **Classification**: The hand skeleton image is processed by a CNN model
5. **Smoothing**: Predictions are buffered for stable output
6. **Translation**: The recognized ASL gesture is displayed

## 📂 Project Structure

- `main.py`: Primary application with real-time ASL translation
- `hand_tracking_cam.py`: Simple hand tracking visualization
- `create_landmark_dataset.py`: CNN model training for skeleton images
- `train_asl_classifier.py`: Training script for the classifier
- `asl_skeleton_cnn_best.h5`: Pre-trained model
- `dataset/`: Training images organized by class (0-9, a-z)

## 🔄 Model Architecture

The CNN model consists of:
- 4 Convolutional layers (32, 64, 128, 256 filters)
- Max Pooling layers
- Dense layers (256 neurons)
- Dropout (0.3) for regularization
- Softmax output for 36 classes

## 📝 License

[MIT License](LICENSE)

## 👨‍💻 Developer

Developed by [L1ght](https://github.com/nour23019870)

## 👏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- [OpenCV](https://opencv.org/) for computer vision processing
