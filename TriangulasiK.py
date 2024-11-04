# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:54:11 2024

@author: visikom2023
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:30:29 2024

@author: visikom2023
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the camera calibration parameters
camera0_params = np.load('camera0_calibration.npz')
camera1_params = np.load('camera1_calibration.npz')
stereo_params = np.load('stereo_calibration.npz')

camera_matrix0 = camera0_params['camera_matrix']
dist_coeffs0 = camera0_params['dist_coeffs']
camera_matrix1 = camera1_params['camera_matrix']
dist_coeffs1 = camera1_params['dist_coeffs']
R = stereo_params['R']
T = stereo_params['T']

# Define the dimensions of the checkerboard
CHECKERBOARD = (8, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initialize webcams
cap0 = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Camera 0
cap1 = cv2.VideoCapture(2,cv2.CAP_DSHOW)  # Camera 1

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open video devices.")
    exit()

# Function to apply 3D rotation
def rotate_3d(points, axis, angle):
    angle_rad = np.deg2rad(angle)
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        R = np.eye(3)
    
    return np.dot(R, points)

iteration = 0

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    
    if not ret0 or not ret1:
        print("Error: Could not read frames.")
        break

    # Convert to grayscale
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)

    if ret0 and ret1:
        # Refine corner locations
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        
        # Draw and display the corners
        frame0_with_corners = cv2.drawChessboardCorners(frame0.copy(), CHECKERBOARD, corners0, ret0)
        frame1_with_corners = cv2.drawChessboardCorners(frame1.copy(), CHECKERBOARD, corners1, ret1)
        
        # Concatenate the frames horizontally
        combined_frame_with_corners = np.hstack((frame0_with_corners, frame1_with_corners))
        
        # Display the combined frame with corners
        cv2.imshow('Chessboard Corners', combined_frame_with_corners)
        
        # Convert corners to the correct format for triangulation
        p0 = np.array(corners0).reshape(-1, 2).T  # Shape (2, n)
        p1 = np.array(corners1).reshape(-1, 2).T  # Shape (2, n)
        
        # Compute projection matrices
        P0 = np.dot(camera_matrix0, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P1 = np.dot(camera_matrix1, np.hstack((R, T.reshape(3, 1))))
        
        # Perform triangulation
        points_4d = cv2.triangulatePoints(P0, P1, p0, p1)
        
        # Convert homogeneous coordinates to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        # Apply 3D rotations
      #  points_3d = rotate_3d(points_3d, 'z', 45)
      #  points_3d = rotate_3d(points_3d, 'x', -60)
        
        # Plot the points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[0], points_3d[1], points_3d[2])
        
        # Initialize plot limits
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Show the plot
        plt.show(block=False)
        plt.pause(0.001)
        
        # Save the plot
        plt.savefig(f'3d_plot.png')
        plt.close()
        
        iteration += 1
        
    else:
        # If corners are not found in both frames, display the combined grayscale images
        combined_gray = np.hstack((gray0, gray1))
        cv2.imshow('Chessboard Corners', combined_gray)
        
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()