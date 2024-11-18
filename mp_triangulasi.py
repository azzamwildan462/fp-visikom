import cv2
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import mediapipe as mp

# =======================================================================================================

class LowPassFilter:
    def __init__(self, alpha=1):
        self.alpha = alpha  # Smoothing factor (0 < alpha <= 1)
        self.prev_value = None  # Stores the previous filtered value

    def filter(self, value):
        if self.prev_value is None or self.prev_value.shape != value.shape:
            # Reset filter if it's the first value or the shape changes
            self.prev_value = value
        else:
            # Apply the EMA formula
            self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        return self.prev_value



lpf = LowPassFilter(alpha=0.1)
# =======================================================================================================


# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.7)

POSE_LANDMARKS = {
    0: 'HIDUNG', 
    11: 'BAHU_KIRI', 
    12: 'BAHU_KANAN', 
    13: 'SIKU_KIRI', 
    14: 'SIKU_KANAN', 
    15: 'PERGELENGAN_TANGAN_KIRI', 
    16: 'PERGELENGAN_TANGAN_KANAN', 
    23: 'PINGGUL_KIRI', 
    24: 'PINGGUL_KANAN', 
    25: 'LUTUT_KIRI', 
    26: 'LUTUT_KANAN', 
    27: 'PERGELENGAN_KAKI_KIRI', 
    28: 'PERGELENGAN_KAKI_KANAN', 
}

# =======================================================================================================

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

# Function to extract 2D landmark positions
def get_landmarks(frame, pose_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb_frame)
    
    if result.pose_landmarks:
        landmarks = {}
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            if landmark.visibility < 0.5:
                continue
            landmarks[idx] = (landmark.x, landmark.y, landmark.z)  # Store x, y, z coordinates
        return landmarks
    else:
        return None

def clean_landmarks(landmarks0, landmarks1):
    if landmarks0 is None or landmarks1 is None:
        return None, None
    
    # Filter out landmarks that are not detected in both frames
    idx_detected_0 = list(landmarks0.keys())
    idx_detected_1 = list(landmarks1.keys())

    # Filter index to just using index list on POSE_LANDMARKS 
    idx_detected_0 = [idx for idx in idx_detected_0 if idx in POSE_LANDMARKS]
    idx_detected_1 = [idx for idx in idx_detected_1 if idx in POSE_LANDMARKS]

    landmarks0 = {idx: landmarks0[idx] for idx in idx_detected_0}
    landmarks1 = {idx: landmarks1[idx] for idx in idx_detected_1}

    return landmarks0, landmarks1

# Ensure both frames have detected the same landmarks at the same indices
def check_landmarks_match(landmarks0, landmarks1):
    if landmarks0 is None or landmarks1 is None:
        return False
    
    # Check if both landmarks have the same indices and are detected in both frames
    for idx in POSE_LANDMARKS:
        if idx not in landmarks0 or idx not in landmarks1:
            print(f"Landmark {POSE_LANDMARKS[idx]} not detected at index {idx} in both frames.")
            return False
    return True

# Function to triangulate 2D points
def triangulate_points(landmarks0, landmarks1):
    points3D = []

    # Compute projection matrices
    P0 = np.dot(camera_matrix0, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1 = np.dot(camera_matrix1, np.hstack((R, T.reshape(3, 1))))

    p0_list = []
    p1_list = []

    idx_to_remove = []

    # Loop over the landmarks and triangulate corresponding points
    for idx in landmarks0:
        if idx in landmarks1:
            # Get 2D points from both frames
            p0 = np.array(landmarks0[idx][:2])  # x, y from frame0
            p1 = np.array(landmarks1[idx][:2])  # x, y from frame1

            p0[0] *= 1280
            p0[1] *= 720
            p1[0] *= 1280
            p1[1] *= 720

            if(p0[0] < 1280 and p0[1] < 720 and p1[0] < 1280 and p1[1] < 720 and p0[0] > 0 and p0[1] > 0 and p1[0] > 0 and p1[1] > 0):
                p0_list.append(p0)
                p1_list.append(p1)
            else:
                idx_to_remove.append(idx)
    
    # print(f'Idx to remove: {idx_to_remove}')

    # Remove the indices that are out of bounds
    for idx in idx_to_remove:
        del landmarks0[idx]
        del landmarks1[idx]

    # Convert to numpy arrays
    p0_list = np.array(p0_list).T
    p1_list = np.array(p1_list).T

    # print(f'{p0_list}')
    # print("================")
    # print(f'{p1_list}')

    # Print all shape 
    # print(f'{P0.shape} {P1.shape} {p0_list.shape} {p1_list.shape}')

    # Triangulate the points
    try:
        point_4d = cv2.triangulatePoints(P0, P1, p0_list, p1_list)
    except Exception as e:
        print(f'Trianulation error: {e}')
        return None, None, None
    
    # Convert back to non-homogeneous coordinates (x, y, z)
    point_3d = point_4d[:3] / point_4d[3]  # Homogeneous to non-homogeneous

    # Apply low-pass filter
    smoothed_point_3d = lpf.filter(point_3d)
    
    return np.array(smoothed_point_3d), landmarks0, landmarks1

# ======================================== INT MAIN ====================================================

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"

# Auto reload
plt.ion()

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

# Process storage
os.makedirs('mp_triang', exist_ok=True)
files_mp_triang = glob.glob('mp_triang/*')
for f in files_mp_triang:
    os.remove(f)

# Open camera
cap = cv2.VideoCapture("/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
cap.set(cv2.CAP_PROP_FPS, 30)

# Plot the points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter([], [], [])

# # Initialize plot limits
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.set_zlim([0, 100])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

iteration = 0
is_not_showed = True
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue
    
    # Membagi frame menjadi dua bagian karena kamera stereonya seperti itu
    frame1 = frame[:, :1280]
    frame0 = frame[:, 1280:]

    frame0 = cv2.flip(frame0, 1)
    frame1 = cv2.flip(frame1, 1)

    # Get landmarks for both frames
    landmarks0 = get_landmarks(frame0, pose)
    landmarks1 = get_landmarks(frame1, pose)

    # Clean landmarks by removing indices that don't match
    landmarks0, landmarks1 = clean_landmarks(landmarks0, landmarks1)

    # Ensure there are valid landmarks for triangulation
    if landmarks0 and landmarks1:
        # Triangulate the corresponding points
        points_3d, landmarks0, landmarks1 = triangulate_points(landmarks0, landmarks1)

        # if landmarks0 and landmarks1 are None then continue
        if points_3d is None:
            continue

        # WTF safety keys check, must be same
        flag_not_same = False
        for idx in landmarks0:
            if idx not in landmarks1:
                flag_not_same = True
                break
        
        if flag_not_same:
            continue

        # Draw landmarks0 and landmarks1 on the frame
        for idx in landmarks0:
            x0, y0, _ = landmarks0[idx]
            x1, y1, _ = landmarks1[idx]
            cv2.circle(frame0, (int(x0 * 1280), int(y0 * 720)), 5, (0, 255, 0), -1)
            cv2.circle(frame1, (int(x1 * 1280), int(y1 * 720)), 5, (0, 255, 0), -1)
        
        # print(f'Points 3D: {points_3d}')
        # print("=====================================")

        # # Plot the points
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # Convert points_3d[*] multiply by 0.1
        points_3d[0] = points_3d[0] * 0.1 * -1
        points_3d[1] = points_3d[1] * 0.1
        points_3d[2] = points_3d[2] * 0.1 * -1 

        # Get mean of points_3d[2]
        mean_z = np.mean(points_3d[2])
        # print(f'It: {iteration} || Mean Z: {mean_z}')


        # idx_to_remove = []
        # for i in range(points_3d.shape[1]):
        #     if points_3d[2][i] < 0:
        #         idx_to_remove.append(i)
        
        # Remove the indices that are out of bounds
        # try:
        #     for idx in idx_to_remove:
        #         points_3d = np.delete(points_3d, idx, axis=1)
        # except Exception as e:
        #     print(f'Error: {e}')
        

        # Print x,y,z index 
        for i in range(points_3d.shape[1]):
            print(f'Point {i}: {points_3d[:, i]}')

        # points_3d[1] = 0
        
        # ax.scatter(points_3d[2], points_3d[0], points_3d[1])

        # # Initialize plot limits
        # ax.set_xlim([100, 300])
        # ax.set_ylim([-100, 200])
        # ax.set_zlim([0, 100])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # # # Show the plot
        # if is_not_showed:
        # plt.show()
            # is_not_showed = False
        # plt.pause(1)
        # plt.close()

        # # Save based on iteration 
        # plt.savefig(f'hasil_plot/3d_plot_{iteration}.png')
        # plt.close()
        
        iteration += 1
        
    combined_gray = np.hstack((frame0, frame1))
    resized_combined_gray = cv2.resize(combined_gray, None, fx=0.5, fy=0.5)
    cv2.imshow('Mediapipe raw', resized_combined_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
