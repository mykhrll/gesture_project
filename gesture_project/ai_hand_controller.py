import cv2
import mediapipe as mp
import pyautogui
import time

# ------------------------------
# KONFIGURASI
# ------------------------------

pyautogui.FAILSAFE = False

cooldown = 0.5
last_action = 0
previous_fingers = -1

gesture_buffer = []
buffer_size = 5

# ------------------------------
# MEDIAPIPE
# ------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ------------------------------
# KAMERA
# ------------------------------

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

print("AI FINGER GESTURE CONTROLLER AKTIF")

# ------------------------------
# LOOP PROGRAM
# ------------------------------

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:

        for hand_lms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = img.shape
            lm_list = []

            for id, lm in enumerate(hand_lms.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lm_list.append((id, cx, cy))

            fingers = []

            finger_tips = [8,12,16,20]
            finger_pip = [6,10,14,18]

            for tip, pip in zip(finger_tips, finger_pip):

                if lm_list[tip][2] < lm_list[pip][2]:

                    fingers.append(1)

                    cv2.circle(img,(lm_list[tip][1],lm_list[tip][2]),10,(0,255,0),cv2.FILLED)

                else:

                    fingers.append(0)

                    cv2.circle(img,(lm_list[tip][1],lm_list[tip][2]),10,(0,0,255),cv2.FILLED)

            total_fingers = sum(fingers)

    # ------------------------------
    # STABILISASI GESTURE
    # ------------------------------

    gesture_buffer.append(total_fingers)

    if len(gesture_buffer) > buffer_size:
        gesture_buffer.pop(0)

    if gesture_buffer.count(total_fingers) == buffer_size:

        current_time = time.time()

        if total_fingers != previous_fingers and current_time - last_action > cooldown:

            if total_fingers == 1:

                print("KANAN")
                pyautogui.press("right")

            elif total_fingers == 2:

                print("KIRI")
                pyautogui.press("left")

            elif total_fingers == 3:

                print("LOMPAT")
                pyautogui.press("up")

            elif total_fingers == 4:

                print("SLIDE")
                pyautogui.press("down")

            previous_fingers = total_fingers
            last_action = current_time

    # ------------------------------
    # CROP KAMERA 1:1
    # ------------------------------

    h, w, _ = img.shape
    size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    img_square = img[start_y:start_y+size, start_x:start_x+size]

    img_square = cv2.resize(img_square,(500,500))

    # ------------------------------
    # UI (DITAMPILKAN SETELAH CROP)
    # ------------------------------

    cv2.rectangle(img_square,(0,0),(300,80),(0,0,0),-1)

    cv2.putText(
        img_square,
        f"Jumlah Jari: {total_fingers}",
        (10,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("AI Finger Gesture Controller", img_square)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()