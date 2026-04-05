import time

import cv2


class FaceProximityDetector:
    def __init__(self, default_target_w=150):
        """
        初始化测距模块
        :param default_target_w: 默认的触发像素宽度，默认值为150px
        """
        self.default_target_w = default_target_w

        # 初始化模型池，避免在每一帧处理时重复加载模型。
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"无法加载 Haar 模型: {cascade_path}")

    def process_frame(self, frame, custom_target_w=None):
        """
        处理单帧图像，输出触发信号
        :param frame: 摄像头读取的单帧图像
        :param custom_target_w: 供外部动态修改触发阈值。如果不传，则使用默认值
        :return: is_triggered
                 bool 信号，距离达标时为 True
        """
        active_target_w = custom_target_w if custom_target_w is not None else self.default_target_w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return False

        # 只关心最靠近镜头的人脸。
        largest_face = max(faces, key=lambda rect: rect[2])
        _, _, w, _ = largest_face
        is_triggered = w >= active_target_w

        return is_triggered


def wait_for_face_trigger(
    camera_index=0,
    target_width=150,
    read_fail_sleep=0.1,
    flush_frame_count=5,
    cooldown=3.0,
    log=True,
):
    """
    打开摄像头并持续检测人脸距离，达到阈值后返回触发信号。

    :param camera_index: 摄像头索引
    :param target_width: 人脸宽度触发阈值
    :param read_fail_sleep: 抓帧失败后的等待时间
    :param flush_frame_count: 触发后清理缓存帧数量
    :param cooldown: 触发后的防抖时间
    :param log: 是否打印日志
    :return: is_triggered
             bool，达到阈值时为 True
    """
    if log:
        print(f"[System] 初始化摄像头 /dev/video{camera_index} ...")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头索引 {camera_index}，请检查 /dev/video 节点！")

    detector = FaceProximityDetector(default_target_w=target_width)
    if log:
        print("[System] 视觉检测模块已加载，等待用户靠近...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if log:
                    print("[Warn] 抓帧失败，稍后重试...")
                time.sleep(read_fail_sleep)
                continue

            is_triggered = detector.process_frame(frame)
            if not is_triggered:
                continue

            if log:
                print("[SIGNAL] 触发！用户已到达最佳交互距离。")

            time.sleep(cooldown)
            for _ in range(flush_frame_count):
                cap.read()
            return True
    finally:
        cap.release()
        if log:
            print("[System] 摄像头已释放。")
