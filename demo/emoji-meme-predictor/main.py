import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi
SMILE_THRESHOLD = 0.35
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smile_and_thumbs_emoji = cv2.imread("images/jempol.png")
    straight_face_emoji = cv2.imread("images/datar.png")
    hands_up_emoji = cv2.imread("images/waduh.png")

    if smile_and_thumbs_emoji is None:
        raise FileNotFoundError("jempol.png not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("datar.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("waduh.png not found")

    # Resize emojis
    smile_and_thumbs_emoji = cv2.resize(smile_and_thumbs_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Gagal memuat gambar emoji!")
    print(f"Detail kesalahan: {e}")
    print("\nFile yang dibutuhkan:")
    print("- jempol.png (emoji senyum dan jempol)")
    print("- datar.png (emoji wajah datar)")
    print("- waduh.png (emoji angkat tangan)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Sticker Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Webcam Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Window', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Sticker Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Webcam Window', 100, 100)
cv2.moveWindow('Sticker Output', WINDOW_WIDTH + 150, 100)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "WAJAH_DATAR"

        # Check for one hand up
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Deteksi "waduh": salah satu pergelangan tangan lebih tinggi (y lebih kecil) dari hidung
            if (left_wrist.y < nose.y) or (right_wrist.y < nose.y):
                current_state = "ANGKAT_TANGAN"
        
        # Check facial expression if hands not up
        if current_state != "ANGKAT_TANGAN":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    nose_tip = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]

                    # Normalisasi terhadap tinggi wajah
                    face_height = ((chin.x - nose_tip.x)**2 + (chin.y - nose_tip.y)**2)**0.5

                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5

                    if face_height > 0 and mouth_width > 0:
                        mouth_aspect_ratio = (mouth_height / mouth_width) / face_height
                    else:
                        mouth_aspect_ratio = 0

                    # Smoothing antar frame
                    SMOOTHING_ALPHA = 0.4
                    if "prev_mar" not in locals():
                        prev_mar = mouth_aspect_ratio
                    mouth_aspect_ratio = (SMOOTHING_ALPHA * mouth_aspect_ratio) + ((1 - SMOOTHING_ALPHA) * prev_mar)
                    prev_mar = mouth_aspect_ratio

                    # Deteksi jempol naik
                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark

                        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                        left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
                        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                        right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]

                        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                        body_height = abs(shoulder.y - hip.y)

                        # jempol boleh di depan wajah atau dada
                        # ambil tinggi wajah sebagai batas area depan wajah
                        head_y = nose_tip.y
                        chest_y = shoulder.y + 0.2 * body_height  # sedikit di bawah bahu

                        # tangan dianggap dalam area "depan wajah/dada"
                        left_in_front = left_wrist.y > head_y - 0.15 and left_wrist.y < chest_y + 0.1
                        right_in_front = right_wrist.y > head_y - 0.15 and right_wrist.y < chest_y + 0.1

                        # jempol tetap harus lebih tinggi dari pergelangan
                        left_thumb_up = left_thumb.y < left_wrist.y
                        right_thumb_up = right_thumb.y < right_wrist.y

                        # kondisi kombinasi
                        thumbs_up = (left_in_front and left_thumb_up) or (right_in_front and right_thumb_up)
                    else:
                        thumbs_up = False

                    if mouth_aspect_ratio > SMILE_THRESHOLD and thumbs_up:
                        current_state = "SENYUM_DAN_JEMPOL"
                    else:
                        current_state = "WAJAH_DATAR"


        # Select emoji based on state
        if current_state == "SENYUM_DAN_JEMPOL":
            emoji_to_display = smile_and_thumbs_emoji
            emoji_name = "😊"
        elif current_state == "WAJAH_DATAR":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        elif current_state == "ANGKAT_TANGAN":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        cv2.putText(camera_frame_resized, f'Kondisi: {current_state}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Tekan "q" untuk keluar', (10, WINDOW_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Window', camera_frame_resized)
        cv2.imshow('Sticker Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
