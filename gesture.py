import cv2
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

GESTURE_LABELS = {
    "OK":          ("OK",         (0, 200, 100)),
    "POINT_LEFT":  ("← 向左指",  (255, 150,   0)),
    "POINT_RIGHT": ("→ 向右指",  (255, 150,   0)),
    "OTHER":       ("...",        (180, 180, 180)),
    "NO_HAND":     ("无手势",     (100, 100, 100)),
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

print("=== 手势识别启动，按 q 退出 ===")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 水平翻转，镜像更直觉
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO_HAND"
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            # 画关键点和连线
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )
        lm = result.multi_hand_landmarks[0].landmark
        gesture = classify(lm)

    label, color = GESTURE_LABELS[gesture]

    # 顶部半透明黑色背景条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 手势文字
    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)

    # 右上角小提示
    cv2.putText(frame, "press q to quit", (430, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()