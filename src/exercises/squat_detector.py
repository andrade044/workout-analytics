import cv2
import mediapipe as mp
import math


class SquatDetector:
    """Class to detect squats in a video."""

    def __init__(
        self,
        check: bool = True,
        counter: int = 0,
        detector_conf: float = 0.5,
        traking_conf: float = 0.5,
        color_points: tuple = (0, 0, 255),
        color_conections: tuple = (255, 255, 255),
    ):
        """Initialize the squat detector.

        Args:
        ----
            check (bool, optional): Whether to check for the squat position. Defaults to True
            counter (int, optional): The initial count of squats. Defaults to 0
            detector_conf (float, optional): Min confidence for the pose detection. Defaults to 0.5
            traking_conf (float, optional): Min confidence for the pose tracking. Defaults to 0.5
            color_points (tuple, optional): Color for the points. Defaults to (0, 0, 0)
            color_conections (tuple, optional): Color for the connections. Defaults to (255, 255, 255)

        Returns:
        -------
            None

        """
        self.check = check
        self.counter = counter
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

    def calculate_angle(self, a: list, b: list, c: list) -> float:
        """Calculate the angle between three points.

        Args:
        ----
            a (list): The first point in the format [id, x, y]
            b (list): The second point in the format [id, x, y]
            c (list): The third point in the format [id, x, y]

        Returns:
        -------
            float: The angle between the three points

        """
        hip_x, hip_y = a[1], a[2]
        knee_x, knee_y = b[1], b[2]
        ankle_x, ankle_y = c[1], c[2]

        # vectors
        ba = [hip_x - knee_x, hip_y - knee_y]
        bc = [ankle_x - knee_x, ankle_y - knee_y]

        # dot product
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]

        # magnitude of vectors
        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])

        if mag_ba == 0 or mag_bc == 0:
            return 0

        # angle calculation
        angle = math.degrees(math.acos(dot_product / (mag_ba * mag_bc)))

        return angle

    def count_squats(
        self, list_points: list, frame: cv2.Mat, down_angle: int = 80, up_angle: int = 160
    ) -> int:
        """Count the number of squats in the given list of pose landmarks.

        Args:
        ----
            list_points (list): A list of the pose landmarks in the format [id, x, y]
            frame (cv2.Mat): The frame to draw the squat count on
            down_angle (int, optional): Angle threshold for the squat down position. Defaults to 80
            up_angle (int, optional): The angle threshold for the squat up position. Defaults to 160

        Returns:
        -------
            int: The number of squats detected

        """
        if list_points:
            LEFT_HIP_IDX = 23
            LEFT_KNEE_IDX = 25
            LEFT_ANKLE_IDX = 27

            left_hip = list_points[LEFT_HIP_IDX]
            left_knee = list_points[LEFT_KNEE_IDX]
            left_ankle = list_points[LEFT_ANKLE_IDX]

            angle = self.calculate_angle(left_hip, left_knee, left_ankle)

            if self.check and angle <= down_angle:
                self.counter += 1
                self.check = False
            if angle >= up_angle:
                self.check = True

            cv2.putText(
                frame,
                f"Squats: {self.counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        return self.counter
