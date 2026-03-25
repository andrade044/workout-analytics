import cv2
import mediapipe as mp


class SquatDetector:
    """Class to detect squats in a video."""

    def __init__(
        self,
        detector_conf: float = 0.5,
        traking_conf: float = 0.5,
        color_points: tuple = (0, 0, 255),
        color_conections: tuple = (255, 255, 255),
    ):
        """Initialize the squat detector.

        Args:
        ----
            detector_conf (float, optional): Min confidence for the pose detection. Defaults to 0.5
            traking_conf (float, optional): Min confidence for the pose tracking. Defaults to 0.5
            color_points (tuple, optional): Color for the points. Defaults to (0, 0, 0)
            color_conections (tuple, optional): Color for the connections. Defaults to (255, 255, 255)

        Returns:
        -------
            None

        """
        self.detector_conf = detector_conf
        self.tracking_conf = traking_conf
        self.color_points = color_points

        self.color_conections = color_conections

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.detector_conf,
            min_tracking_confidence=self.tracking_conf,
            model_complexity=1,
            smooth_landmarks=True,
        )

        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_drawing_config_points = self.mp_drawing.DrawingSpec(color=self.color_points)

        self.mp_drawing_config_conections = self.mp_drawing.DrawingSpec(
            color=self.color_conections,
        )

    def detect(self, frame: cv2.Mat, drawing: bool = True) -> cv2.Mat:
        """Detect the pose in the given frame.

        Args:
        ----
            frame (cv2.Mat): The frame to detect the pose in
            drawing (bool, optional): Whether to draw the pose landmarks and connections

        Returns:
        -------
            cv2.Mat: The frame with the detected pose.

        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(frame_rgb)

        if self.result.pose_landmarks.landmark:
            if drawing:
                self.mp_drawing.draw_landmarks(
                    frame,
                    self.result.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing_config_points,
                    self.mp_drawing_config_conections,
                )

        return frame

    def find_points(self, frame: cv2.Mat) -> list:
        """Find the pose landmarks in the given frame.

        Args:
        ----
            frame (cv2.Mat): The frame to find the pose landmarks in

        Returns:
        -------
            list: A list of the pose landmarks in the format [id, x, y]

        """
        list_points = []
        if self.result.pose_landmarks.landmark:
            for id, points in enumerate(self.result.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(points.x * w), int(points.y * h)

                list_points.append([id, cx, cy])

        return list_points
