import cv2
import mediapipe as mp 


class SquatDetector:
    def __init__(self, detector_conf=0.5,
                 traking_conf=0.5, color_points=(0,0,255,), color_conections=(255,255,255)
                 ):

        self.detector_conf=detector_conf
        self.tracking_conf=traking_conf
        self.color_points=color_points
        self.color_conections=color_conections
        
        

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=self.detector_conf,
                                    min_tracking_confidence=self.tracking_conf, model_complexity=1,
                                    smooth_landmarks=True)     


        self.mp_drawing = mp.solutions.drawing_utils  

        self.mp_drawing_config_points = self.mp_drawing.DrawingSpec(color=self.color_points)

        self.mp_drawing_config_conections = self.mp_drawing.DrawingSpec(color=self.color_conections, )



    def detect(self, frame, drawing=True):          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.result = self.pose.process(frame_rgb)
        print(self.result)

        if self.result.pose_landmarks.landmark:
            if drawing:
                self.mp_drawing.draw_landmarks(frame, 
                                                self.result.pose_landmarks,
                                                self.mp_pose.POSE_CONNECTIONS,
                                                self.mp_drawing_config_points, 
                                                self.mp_drawing_config_conections) 

        return frame
    

    def find_points(self, frame, drawing=True):
        list_points = []
        if self.result.pose_landmarks.landmark:

           for id, points in enumerate(self.result.pose_landmarks.landmark):
                h ,w,c = frame.shape
                cx, cy = int(points.x * w), int(points.y * h)

                list_points.append([id, cx, cy])

        return list_points