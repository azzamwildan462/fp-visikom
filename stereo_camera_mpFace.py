import cv2
import mediapipe as mp

# Initialize MediaPipe Pose Mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Pose Mesh with static_image_mode=False for video
pose_mesh = mp_pose.Pose(
    static_image_mode=False,  # Set to False for video input so it tracks over frames
    model_complexity=1,       # 0, 1, or 2. Higher values for more accurate landmark points
    smooth_landmarks=True,    # Smooth landmarks to avoid jitter in video processing
    enable_segmentation=False, # Set to True if you want segmentation (background removal)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmark_indices = [0, 1, 2, 3, 4, 11, 12, 13, 14]
lm_kiri = []
lm_kanan = []

korespondensi_buffer = []

def mpLm2px(lm):
    return (lm.x * 1280), (lm.y * 720)

def process_mp(frame_id,frame): 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for pose landmarks detection
    result = pose_mesh.process(frame_rgb)

    global lm_kiri 
    global lm_kanan

    if frame_id == 0:
        lm_kiri = []
    elif frame_id == 1:
        lm_kanan = []

    # Draw pose landmarks on the frame
    if result.pose_landmarks:
        for idx in landmark_indices:
            lm_1 = result.pose_landmarks.landmark[idx]
            px, py = mpLm2px(lm_1)

            # Store landmarks for left and right frames
            if frame_id == 0:
                lm_kiri.append((px, py))
            elif frame_id == 1:
                lm_kanan.append((px, py))

            # Draw a green circle on the landmark
            try:
                cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 0), -1)
            except Exception as e:
                print(f"Error drawing circle: {e}")



def process_korespondensi_1_1():
    if len(lm_kiri) == len(lm_kanan) == len(landmark_indices): # Memastikan kedua gambar mendeteksi landmark yang sama
        lm_it = 0

        mp_kiri_filtered = []
        for idx in landmark_indices:
            # Filter from mines x or y 
            if lm_kiri[lm_it][0] < 0 or lm_kanan[lm_it][0] < 0 or lm_kiri[lm_it][1] < 0 or lm_kanan[lm_it][1] < 0:
                lm_it += 1
                continue

            # print(f'pt[{idx}]: {lm_kiri[lm_it]} || {lm_kanan[lm_it]}')

            mp_kiri_filtered.append(lm_kiri[lm_it])

            lm_it += 1
        # print("=====================================")


        print(mp_kiri_filtered)
        print("=====================================")


# Open the webcam
cap = cv2.VideoCapture("/dev/v4l/by-id/usb-SunplusIT_Inc_SPCA2100_PC_Camera-video-index0")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Slice image 
    cam_kiri = frame[:, :1280]
    cam_kanan = frame[:, 1280:]

    process_mp(0,cam_kiri)
    process_mp(1,cam_kanan)
    process_korespondensi_1_1()


    # Process registrasi points 
    # Masih belum... 

    resized_cam_kiri = cv2.resize(cam_kiri, None, fx=0.5, fy=0.5)
    resized_cam_kanan = cv2.resize(cam_kanan, None, fx=0.5, fy=0.5)
    concatenated_image = cv2.hconcat([resized_cam_kiri, resized_cam_kanan]) # h-concate -> horizontal

    # Show the output frame
    # cv2.imshow('Kiri', resized_cam_kiri)
    # cv2.imshow('Kanan', resized_cam_kanan)
    cv2.imshow('concatenated_image', concatenated_image)



    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
