 Emotion-Based Song Recommendation System
 Overview
This project is an AI-powered emotion detection system using Mediapipe, TensorFlow, OpenCV, and Streamlit to recognize facial expressions and recommend songs based on emotions.

Features
- Real-time face and hand landmark detection using Mediapipe.
- Emotion classification using a trained deep learning model.
- Streamlit-based web application with real-time video streaming.
- Automatic song recommendations based on detected emotions and user preferences (language & singer).

 Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For video capture and processing.
- **Mediapipe**: For facial and hand landmark detection.
- **TensorFlow/Keras**: For building and training the emotion classification model.
- **Streamlit**: For creating an interactive web application.
- **NumPy**: For data handling and processing.

How It Works
1. **Collect Data**: Captures face and hand landmark data and stores it in `.npy` format.
2. **Train Model**: Loads the stored landmark data, trains a neural network, and saves the model (`model.h5`).
3. **Real-time Emotion Detection**: The trained model predicts emotions from live video feed.
4. **Song Recommendation**: Based on detected emotion, it generates a YouTube search link for music recommendations.


Usage
1. Enter your preferred language and singer.
2. Allow the application to capture your emotion using the webcam.
3. Click the "Recommend me songs" button to open YouTube with suitable song suggestions.

 Future Improvements
- Enhancing model accuracy with more training data.
- Supporting multiple emotions for better song recommendations.
- Improving UI for a better user experience.



