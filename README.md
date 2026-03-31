# 🏋️ Workout Analytics – Squat Detector

This project explores how computer vision can be applied to fitness.

The idea is to detect and count squats automatically using pose estimation from a video or webcam.

Instead of relying on sensors, the system uses body landmarks to understand movement.

---

## 📌 How it works

The squat detection is based on **joint angle analysis**.

* The system tracks three key points:

  * Hip
  * Knee
  * Ankle

* It calculates the angle at the knee using vector math

* Based on this angle:

  * When the angle is **low (≈ 80°)** → squat down
  * When the angle is **high (≈ 160°)** → standing up

---

## 🚀 Features

* Squat detection using pose estimation
* Repetition counting
* Angle-based movement analysis
* Support for video input
* Modular and extensible structure

---

## 🧠 Technologies

* Python 3.12
* MediaPipe 0.10.21
* OpenCV 4.11
* NumPy 1.26.4

---

## 📂 Project Structure

```
workout-analytics/
│
├── src/
│   └── exercises/
│       └── squat_detector.py
│
├── videos/
│   └── input/
│
├── main.py
├── pyproject.toml
└── README.md
```

---

## ▶️ Getting Started

### Install dependencies

```bash
make install
```

or

```bash
uv sync
```

---

### Run the project

```bash
python main.py
```

---

## 🎥 Demo

<p align="center">
  <img src="assets/demo.gif" width="500"/>
</p>

---

## 📹 Test Videos

Example videos are available in:

```
videos/input/
```

You can use them or replace with your own recordings.

---

## 🔮 Future Improvements

* Posture correction feedback
* Support for other exercises
* Real-time webcam optimization
* Performance improvements
* UI/dashboard

---

## 👨‍💻 Author

Matheus Andrade

* LinkedIn: https://www.linkedin.com/in/matheus-andrade-6b86a9210
* Email: [matheusandrade906@gmail.com](mailto:matheusandrade906@gmail.com)

---

