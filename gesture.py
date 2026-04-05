import cv2
import mediapipe as mp
import math
import time
import os

# 根据 ls /dev/video* 的结果修改这里
CAMERA_INDEX = 0

SAVE_DIR = os.path.expanduser("~/CAT/saved")
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(f"❌ 摄像头 {CAMERA_INDEX} 打开失败，请检查 /dev/video* 设备号")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

GESTURE_LABELS = {
    "OK":          ("OK",        (0, 200, 100)),
    "POINT_LEFT":  ("<- LEFT",   (255, 150,   0)),
    "POINT_RIGHT": ("-> RIGHT",  (255, 150,   0)),
    "OTHER":       ("...",       (180, 180, 180)),
    "NO_HAND":     ("NO HAND",   (100, 100, 100)),
}

def finger_extended(lm, tip, pip_idx):
    return lm[tip].y < lm[pip_idx].y

def dist(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def classify(lm):
    index_ext = finger_extended(lm, 8, 6)
    mid_ext   = finger_extended(lm, 12, 10)
    ring_ext  = finger_extended(lm, 16, 14)
    pinky_ext = finger_extended(lm, 20, 18)
    thumb_index_d = dist(lm, 4, 8)

    if thumb_index_d < 0.07 and mid_ext and ring_ext and pinky_ext:
        return "OK"
    if index_ext and not mid_ext and not ring_ext and not pinky_ext:
        if lm[8].x < lm[0].x:
            return "POINT_LEFT"
        else:
            return "POINT_RIGHT"
    return "OTHER"

print("=== 手势识别启动 | s=保存 q=退出 ===")

last_saved_name = ""
save_tip_until  = 0   # 显示保存提示的截止时间

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO_HAND"
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        lm = result.multi_hand_landmarks[0].landmark
        gesture = classify(lm)

    label, color = GESTURE_LABELS[gesture]

    # 顶部黑色背景条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 手势标签
    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3, cv2.LINE_AA)

    # 右上角操作提示
    cv2.putText(frame, "s=save  q=quit", (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # 底部保存提示（按s后显示3秒）
    if time.time() < save_tip_until:
        cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, f"Saved: {last_saved_name}", (10, 468),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{ts}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        last_saved_name = filename
        save_tip_until  = time.time() + 3.0
        print(f"[保存] {filepath}")

cap.release()
cv2.destroyAllWindows()