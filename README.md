# VirgilDrive

A real-time drowsiness detection app that uses MediaPipe face landmarks and a camera feed to warn when the driver appears sleepy.

## Project structure

- `main.py` — entry point
- `src/drowsiness_detector.py` — detection logic
- `assets/face_landmarker.task` — MediaPipe face landmark model
- `requirements.txt` — Python dependencies

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Press `q` to quit.

## How it works

- Detects face landmarks from your webcam.
- Calculates the Eye Aspect Ratio (EAR).
- If the eyes stay closed for a set period, it triggers an audio alarm.

## GitHub publishing

```bash
git init
git add .
git commit -m "Initial project setup"
git branch -M main
```

Then create a new empty GitHub repository on GitHub.com and run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Notes

- The model file is large and is kept in `assets/` so the project is easier to understand and share.
- Do not commit your local virtual environment folder `.venv`.
