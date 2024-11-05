import cv2
import os
import numpy as np
from datetime import datetime
import time
from pathlib import Path

def load_calibration_data(filename):
    """Memuat parameter kalibrasi dari file .npz."""
    with np.load(filename) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs

def undistort_webcam_feed(camera_matrix, dist_coeffs):
    """Menjalankan webcam dengan koreksi distorsi menggunakan parameter kalibrasi."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    while True:
        # Membaca frame dari webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        # Mengoreksi distorsi pada frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)


        # Menampilkan frame yang sudah dikoreksi distorsinya
        cv2.imshow('Undistorted Webcam Feed', undistorted_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Membersihkan dan menutup
    cap.release()
    cv2.destroyAllWindows()

# Memastikan file kalibrasi tersedia
calibration_file = f"calibration_0.npz"
try:
    camera_matrix, dist_coeffs = load_calibration_data(calibration_file)
    # Menampilkan webcam feed yang sudah di-undistort
    undistort_webcam_feed(camera_matrix, dist_coeffs)
except FileNotFoundError:
    print(f"File kalibrasi {calibration_file} tidak ditemukan. Pastikan telah melakukan kalibrasi kamera.")
