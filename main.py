import cv2
import io
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from fer import FER
from fer.classes import Video

app = Flask(__name)

def analyze_video(video_file, mtcnn):
    video = Video(video_file)
    detector = FER(mtcnn=mtcnn)

    # Output list of dictionaries
    raw_data = video.analyze(detector, display=False)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    return df.to_json(orient='records')

@app.route('/analyze_video', methods=['POST'])
def analyze_video_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'})

    video_file = request.files['video']
    mtcnn = bool(int(request.form.get('mtcnn', 0)))

    if not video_file:
        return jsonify({'error': 'Invalid video file'})

    # Save the video file to a temporary location
    video_path = Path('temp_video.mp4')
    video_file.save(video_path)

    try:
        result = analyze_video(video_path, mtcnn)
        return jsonify({'emotions': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
