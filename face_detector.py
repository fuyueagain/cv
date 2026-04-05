import cv2

class FaceProximityDetector:
    def __init__(self, default_target_w=150):
        """
        初始化测距模块
        :param default_target_w: 默认的触发像素宽度，默认值为150px
        """
        self.default_target_w = default_target_w
        
        # 初始化模型池，避免在每一帧处理时重复加载模型（这是新手常犯的性能瓶颈）
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_frame(self, frame, custom_target_w=None):
        """
        处理单帧图像，输出触发信号
        :param frame: 摄像头读取的单帧图像
        :param custom_target_w: 供外部动态修改触发阈值。如果不传，则使用默认值
        :return: (is_triggered, face_info)
                 is_triggered: bool 信号，距离达标时为 True
                 face_info: tuple (x, y, w, h) 最大人脸坐标，用于后续处理或调试
        """
        # 1. 确定本次运算的基准值
        active_target_w = custom_target_w if custom_target_w is not None else self.default_target_w

        # 2. 图像预处理（转灰度图降低计算量）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 3. 执行推理
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )

        # 没有检测到人脸
        if len(faces) == 0:
            return False, None

        # 4. 核心工程逻辑：防干扰过滤
        # 画面中可能有背景噪点或远处路过的人，我们只关心离镜头最近的人脸（即像素宽度 w 最大的那个框）
        largest_face = max(faces, key=lambda rect: rect[2]) 
        x, y, w, h = largest_face

        # 5. 触发判定
        # 像素宽度越大，说明人离摄像头越近。因此 w >= target_w 时触发
        is_triggered = (w >= active_target_w)

        return is_triggered, largest_face