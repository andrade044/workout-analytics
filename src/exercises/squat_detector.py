import cv2
import mediapipe as mp 


class SquatDetector:
    def __init__(self, modo=False, detector_conf=0.5,
                 rastreio_conf=0.5, cor_points=(0,0,255,), cor_conections=(255,255,255)
                 ):

        self.modo=modo
        self.detector_conf=detector_conf
        self.rastreio_conf=rastreio_conf
        self.cor_points=cor_points
        self.cor_conections=cor_conections
        
        

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.modo,
                                    min_detection_confidence=self.detector_conf,
                                    min_tracking_confidence=self.rastreio_conf, model_complexity=1,
                                    smooth_landmarks=True)     



        self.mp_drawing = mp.solutions.drawing_utils  


        self.mp_drawing_config_points = self.mp_drawing.DrawingSpec(color=self.cor_points)

        self.mp_drawing_config_conections = self.mp_drawing.DrawingSpec(color=self.cor_conections, )




    def detectar(self, frame, drawing=True):          
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
    

    def encontrar_points(self, frame, drawing=True):
        list_points = []
        if self.result.pose_landmarks.landmark:

           for id, points in enumerate(self.result.pose_landmarks.landmark):
                h ,w,c = frame.shape
                cx, cy = int(points.x * w), int(points.y * h)

                list_points.append([id, cx, cy])

        return list_points