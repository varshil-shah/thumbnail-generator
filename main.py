from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
import requests
import os
from dotenv import load_dotenv
from http import HTTPStatus
from typing import List, Tuple

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

app = Flask(__name__)

# Initialize CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "thumbnail-generator"
    }), HTTPStatus.OK


def download_video(video_url: str, save_path: str = 'downloaded_video.mp4') -> str:
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        raise Exception(f"Video download failed: {str(e)}")


def extract_key_frames(video_path: str, interval: int = 5) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Failed to open video file")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (interval * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
                frames.append(frame)
            frame_count += 1

        if not frames:
            raise Exception("No frames extracted from video")

        return frames
    finally:
        cap.release()


def evaluate_frame(frame: np.ndarray) -> Tuple[float, float, float]:
    if frame is None or frame.size == 0:
        return 0.0, 0.0, 0.0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    brightness = np.mean(gray_frame)

    b, g, r = cv2.split(frame.astype("float"))
    colorfulness = np.sqrt(((b - r) ** 2).mean() +
                           ((g - r) ** 2).mean() + ((b - g) ** 2).mean())

    return sharpness, brightness, colorfulness


def select_best_frames(frames: List[np.ndarray], top_n: int = 3) -> List[Tuple[np.ndarray, float, float, float, float]]:
    if not frames:
        raise Exception("No frames provided for selection")

    scores = []
    for frame in frames:
        sharpness, brightness, colorfulness = evaluate_frame(frame)
        score = sharpness * 0.5 + brightness * 0.3 + colorfulness * 0.2
        scores.append((frame, sharpness, brightness, colorfulness, score))

    return sorted(scores, key=lambda x: x[4], reverse=True)[:top_n]


def upload_thumbnail(frame: np.ndarray) -> str:
    temp_thumbnail_path = f'temp_thumbnail_{os.getpid()}.jpg'
    try:
        cv2.imwrite(temp_thumbnail_path, frame)
        response = cloudinary.uploader.upload(temp_thumbnail_path)
        return response['secure_url']
    except Exception as e:
        raise Exception(f"Failed to upload thumbnail: {str(e)}")
    finally:
        if os.path.exists(temp_thumbnail_path):
            os.remove(temp_thumbnail_path)


@app.route('/get-thumbnail', methods=['POST'])
def upload_thumbnail_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), HTTPStatus.BAD_REQUEST

        video_url = data.get('video_url')
        if not video_url:
            return jsonify({"error": "No video URL provided"}), HTTPStatus.BAD_REQUEST

        interval = max(1, int(data.get('interval', 5)))
        top_n = max(1, int(data.get('top_n', 3)))

        video_path = f'downloaded_video_{os.getpid()}.mp4'
        try:
            video_path = download_video(video_url, video_path)
            key_frames = extract_key_frames(video_path, interval)
            best_frames = select_best_frames(key_frames, top_n)

            best_frame = best_frames[0][0]
            thumbnail_url = upload_thumbnail(best_frame)

            return jsonify({
                "thumbnail_url": thumbnail_url,
                "metadata": {
                    "sharpness": best_frames[0][1],
                    "brightness": best_frames[0][2],
                    "colorfulness": best_frames[0][3],
                    "overall_score": best_frames[0][4]
                }
            }), HTTPStatus.OK

        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(debug=True)
