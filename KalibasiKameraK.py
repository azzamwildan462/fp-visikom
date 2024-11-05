# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:25:12 2024

@author: visikom2023
"""

import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (8, 6)
CHECKERBOARD = (8, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the dimensions of the checkerboard

# Calibration part

def Kamera(CHECKERBOARD = (8, 6)):

    cap = cv2.VideoCapture("/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create directories to save images
    os.makedirs('cam0', exist_ok=True)
    os.makedirs('cam1', exist_ok=True)
    
    # Clear the directories
    files_cam0 = glob.glob('cam0/*')
    files_cam1 = glob.glob('cam1/*')
    
    for f in files_cam0:
        os.remove(f)
    
    for f in files_cam1:
        os.remove(f)
        
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints1 = []  # 2d points in image plane for camera 1.
    imgpoints2 = []  # 2d points in image plane for camera 2.
    
    
    image_counter = 0
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints1 = []  # 2d points in image plane for camera 1.
    imgpoints2 = []  # 2d points in image plane for camera 2.
    
    while True:
        ret, frame = cap.read()

        if not ret:
            continue
        
        frame1 = frame[:, :1280]
        frame2 = frame[:, 1280:]
        
        # Concatenate the frames horizontally
        combined_frame = np.hstack((frame1, frame2))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

        cv2.imshow('frame', combined_frame)
    
        if ret1 and ret2:
            objpoints.append(objp)
            
            
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            imgpoints1.append(corners1)
            
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints2.append(corners2)
            
            # Save the images before drawing the corners
            cv2.imwrite(f'cam0/{image_counter}.jpg', frame1)
            cv2.imwrite(f'cam1/{image_counter}.jpg', frame2)
            print(f"Saved pair {image_counter}")
            
            # Draw and display the corners
            frame1_with_corners = cv2.drawChessboardCorners(frame1.copy(), CHECKERBOARD, corners1, ret1)
            frame2_with_corners = cv2.drawChessboardCorners(frame2.copy(), CHECKERBOARD, corners2, ret2)
            
            # Concatenate the frames with corners horizontally
            frame1_with_corners = cv2.flip(frame1_with_corners, 1)
            frame2_with_corners = cv2.flip(frame2_with_corners, 1)
            
            combined_frame_with_corners = np.hstack((frame1_with_corners, frame2_with_corners))
            
            # Add counter text to the combined image with corners
            cv2.putText(combined_frame_with_corners, f'Image count: {image_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the combined frame with corners and counter
            cv2.imshow('Img', combined_frame_with_corners)
            
            image_counter += 1
        else:
            
            frame1 = cv2.flip(frame1, 1)
            frame2 = cv2.flip(frame2, 1)
            combined_frame = np.hstack((frame1, frame2))
            
            # Concatenate the grayscale images horizontally
            
            # Add counter text to the combined grayscale image
            cv2.putText(combined_frame, f'Image count: {image_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display the combined grayscale images
            cv2.imshow('Img', combined_frame)
    
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
def calibrate_camera(image_dir, checkerboard_size):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    
    images = glob.glob(f'{image_dir}/*.jpg')
    gray = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError(f"No valid images found in {image_dir} for calibration.")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print(f"Calibration successful for {image_dir}")
    else:
        print(f"Calibration failed for {image_dir}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, gray.shape[::-1]

def read_points(image_dir, checkerboard_size):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    
    images = glob.glob(f'{image_dir}/*.jpg')
    gray = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError(f"No valid images found in {image_dir} for point reading.")
    
    return objpoints, imgpoints

def perform_stereo_calibration(checkerboard_size):
    print("Calibrating camera 0...")
    camera_matrix0, dist_coeffs0, rvecs0, tvecs0, image_shape0 = calibrate_camera('cam0', checkerboard_size)
    
    print("Calibrating camera 1...")
    camera_matrix1, dist_coeffs1, rvecs1, tvecs1, image_shape1 = calibrate_camera('cam1', checkerboard_size)
    
    if camera_matrix0 is not None and camera_matrix1 is not None:
        # Save individual camera calibration results
        np.savez('camera0_calibration.npz', camera_matrix=camera_matrix0, dist_coeffs=dist_coeffs0, 
                 rvecs=rvecs0, tvecs=tvecs0)
        np.savez('camera1_calibration.npz', camera_matrix=camera_matrix1, dist_coeffs=dist_coeffs1, 
                 rvecs=rvecs1, tvecs=tvecs1)
        
        # Read points from cam0 and cam1 again for stereo calibration
        print("Reading points from camera 0...")
        objpoints0, imgpoints0 = read_points('cam0', checkerboard_size)
        
        print("Reading points from camera 1...")
        objpoints1, imgpoints1 = read_points('cam1', checkerboard_size)
        
        if len(objpoints0) == 0 or len(imgpoints0) == 0 or len(objpoints1) == 0 or len(imgpoints1) == 0:
            raise ValueError("Not enough valid points found for stereo calibration.")
        
        # Perform stereo calibration
        print("Performing stereo calibration...")
        ret, camera_matrix0, dist_coeffs0, camera_matrix1, dist_coeffs1, R, T, E, F = cv2.stereoCalibrate(
            objpoints0, imgpoints0, imgpoints1, camera_matrix0, dist_coeffs0, camera_matrix1, dist_coeffs1, image_shape0,
            criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        if ret:
            print("Stereo calibration successful")
            print("Rotation matrix:", R)
            print("Translation vector:", T)
            
            # Save the stereo calibration results
            np.savez('stereo_calibration.npz', camera_matrix0=camera_matrix0, dist_coeffs0=dist_coeffs0, 
                     camera_matrix1=camera_matrix1, dist_coeffs1=dist_coeffs1, R=R, T=T, E=E, F=F)
        else:
            print("Stereo calibration failed")

# Perform the stereo calibration
# Kamera() 
perform_stereo_calibration(CHECKERBOARD)
