import cv2

from src.exercises.squat_detector import SquatDetector


def main() -> None:
    """Run the squat detector on a video file.

    Returns
    -------
        None

    """
    cap = cv2.VideoCapture("videos/input/video-1.mp4")

    detector = SquatDetector()

    while True:
        ret, frame = cap.read()

        frame = detector.detect(frame)

        list_points = detector.find_points(frame)
        detector.count_squats(list_points, frame)

        cv2.imshow("Squat Detector", frame)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
