import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize Mediapipe Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def plot_body_landmarks(landmarks, ax):
    """Plots body landmarks in 3D."""
    xs, ys, zs = [], [], []
    for landmark in landmarks:
        xs.append(landmark.x)
        ys.append(landmark.y)
        zs.append(landmark.z)

    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.plot(xs, ys, zs, 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def process_video():
    # Open webcam
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and find the body pose
            results = holistic.process(image)

            # Draw landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Plot body landmarks in 3D
            if results.pose_landmarks:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                plot_body_landmarks(results.pose_landmarks.landmark, ax)
                plt.show(block=False)
                plt.pause(0.001)
                plt.close(fig)

            # Show the processed video
            cv2.imshow('Body Landmarks', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video()
