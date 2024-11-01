import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Face Mesh with static_image_mode=False for video, max_num_faces=2 to detect up to 2 faces
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,  # This option provides iris tracking landmarks as well
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmark_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80]
lm_kiri = []
lm_kanan = []

korespondensi_buffer = []

def mpLm2px(lm):
    return (lm.x * 1280), (lm.y * 720)

def process_mp(frame_id,frame): 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for face landmarks detection
    result = face_mesh.process(frame_rgb)

    global lm_kiri 
    global lm_kanan

    if frame_id == 0:
        lm_kiri = []
    elif frame_id == 1:
        lm_kanan = []

    # Draw face landmarks on the frame
    if result.multi_face_landmarks:
        # for face_landmarks in result.multi_face_landmarks:
            # Draw landmarks
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,  # Tesselation (triangular mesh)
            #     landmark_drawing_spec=None,  # Use default drawing spec for landmarks
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            # )

            # Example: Get and print all landmark coordinates for the first face
            # for id, lm in enumerate(face_landmarks.landmark):
            #     ih, iw, _ = frame.shape
            #     x, y = int(lm.x * iw), int(lm.y * ih)
            #     print(f'Landmark {id}: ({x}, {y})')

        for idx in landmark_indices:
            lm_1 = result.multi_face_landmarks[0].landmark[idx]    
            px, py = mpLm2px(lm_1)

            if frame_id == 0:
                lm_kiri.append((int(px),int(py)))
                # print(f"Landmark {idx} Kiri: (x: {int(px)}, y: {int(py)})")
            elif frame_id == 1:
                lm_kanan.append((int(px),int(py)))
                # print(f"Landmark {idx} Kanan: (x: {int(px)}, y: {int(py)})")

            try:
                cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 0), -1)  # Draw a green circle
            except Exception as e:
                print(f"Error drawing circle: {e}")



def process_korespondensi_1_1():
    if len(lm_kiri) == len(lm_kanan) == len(landmark_indices): # Memastikan kedua gambar mendeteksi landmark yang sama
        lm_it = 0
        for idx in landmark_indices:
            print(f'if[{idx}]: {lm_kiri[lm_it]} || {lm_kanan[lm_it]}')
            lm_it += 1


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
