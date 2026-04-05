# RDK协作

python3 -c "import cv2; cv2.namedWindow('test'); print('有界面版 OK'); cv2.destroyAllWindows()"



sunrise@ubuntu:~/CAT$ python3 gesture.py
[ WARN:0@2.776] global cap_v4l.cpp:913 open VIDEOIO(V4L2:/dev/video1): can't open camera by index
[ERROR:0@2.862] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
❌ 摄像头 1 打开失败，请检查 /dev/video* 设备号
sunrise@ubuntu:~/CAT$ 
]

sunrise@ubuntu:~/CAT$ sudo python3 gesture.py 
Traceback (most recent call last):
  File "/home/sunrise/CAT/gesture.py", line 2, in <module>
    import mediapipe as mp
ModuleNotFoundError: No module named 'mediapipe'



sunrise@ubuntu:~/CAT$ sudo -E python3 gesture.py 
[ WARN:0@2.719] global cap_v4l.cpp:913 open VIDEOIO(V4L2:/dev/video1): can't open camera by index
[ERROR:0@2.804] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
❌ 摄像头 1 打开失败，请检查 /dev/video* 设备号
sunrise@ubuntu:~/CAT$ 
