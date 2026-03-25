import cv2
from src.exercises.squat_detector import SquatDetector


def main():
    
    cap = cv2.VideoCapture("videos/input/video-1.mp4") 

    detector = SquatDetector()

    while True:
        ret, frame = cap.read()

        frame = detector.detectar(frame)

        list_points = detector.find_points(frame)
        print(list_points)

        cv2.imshow('Squat Detector', frame) 

        cv2.waitKey(1)




if __name__ == "__main__":
    main()
