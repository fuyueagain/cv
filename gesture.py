import math
import time
from typing import Any, Literal, Optional

import cv2
import mediapipe as mp


TargetGesture = Literal["OK", "POINT_LEFT", "POINT_RIGHT"]


OK: TargetGesture = "OK"
POINT_LEFT: TargetGesture = "POINT_LEFT"
POINT_RIGHT: TargetGesture = "POINT_RIGHT"


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_INPUT = True
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
POLL_INTERVAL_SECONDS = 0.01


_cap: Optional[Any] = None
_hands: Optional[Any] = None


def _finger_extended(lm, tip: int, pip_idx: int) -> bool:
    return lm[tip].y < lm[pip_idx].y


def _dist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def _classify(lm) -> Optional[TargetGesture]:
    index_ext = _finger_extended(lm, 8, 6)
    mid_ext = _finger_extended(lm, 12, 10)
    ring_ext = _finger_extended(lm, 16, 14)
    pinky_ext = _finger_extended(lm, 20, 18)
    thumb_index_d = _dist(lm, 4, 8)

    if thumb_index_d < 0.07 and mid_ext and ring_ext and pinky_ext:
        return OK
    if index_ext and not mid_ext and not ring_ext and not pinky_ext:
        if lm[8].x < lm[0].x:
            return POINT_LEFT
        return POINT_RIGHT
    return None


def _create_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def _initialize_resources() -> None:
    global _cap, _hands

    if _cap is None:
        _cap = cv2.VideoCapture(CAMERA_INDEX)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not _cap.isOpened():
            release_camera()
            raise RuntimeError(
                f"无法打开摄像头索引 {CAMERA_INDEX}，请检查 /dev/video*"
            )

    if _hands is None:
        _hands = _create_hands()


def _detect_target_gesture(frame) -> Optional[TargetGesture]:
    if frame is None:
        return None

    if _hands is None:
        raise RuntimeError("手势检测器尚未初始化。")

    working_frame = cv2.flip(frame, 1) if MIRROR_INPUT else frame
    rgb = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
    result = _hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    return _classify(landmarks)


def detect_gesture(timeout: Optional[float] = None) -> TargetGesture:
    """
    阻塞式手势检测函数。

    行为：
    - 首次调用时自动初始化摄像头和 MediaPipe Hands。
    - 持续循环检测，直到识别到 OK / POINT_LEFT / POINT_RIGHT 之一。
    - 检测过程中对未检测到手部、其他手势、手势不明确均静默忽略。

    参数：
    - timeout: 最大等待秒数。超时后抛出 TimeoutError。

    返回值：
    - "OK" | "POINT_LEFT" | "POINT_RIGHT"
    """

    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be greater than 0")

    _initialize_resources()
    deadline = None if timeout is None else time.monotonic() + timeout

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"在 {timeout:.2f} 秒内未检测到 OK / POINT_LEFT / POINT_RIGHT"
            )

        if _cap is None:
            raise RuntimeError("摄像头尚未初始化。")

        ret, frame = _cap.read()
        if not ret:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        gesture = _detect_target_gesture(frame)
        if gesture is not None:
            return gesture

        time.sleep(POLL_INTERVAL_SECONDS)


def release_camera() -> None:
    """
    释放摄像头和 MediaPipe Hands 检测器资源。
    """

    global _cap, _hands

    if _cap is not None:
        _cap.release()
        _cap = None

    if _hands is not None:
        _hands.close()
        _hands = None


if __name__ == "__main__":
    try:
        print(detect_gesture())
    finally:
        release_camera()
