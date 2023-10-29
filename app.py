import cv
import io
import os
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from fer import FER
from fer.classes import Video
from concurrent.futures import ThreadPoolExecutor
import uuid
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor()

# Create a dictionary to track task statuses and results
task_results = {}

def analyze_video(video_file, mtcnn, task_id):
    video = Video(str(video_file))  # Convert video_path to a string
    detector = FER(mtcnn=mtcnn)

    # Output list of dictionaries
    raw_data = video.analyze(detector, display=False)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    # Store the result in the task_results dictionary
    task_results[task_id] = df.to_dict(orient='records')

    # Delete the input video file after analysis is completed
    try:
        os.remove(video_file)
    except Exception as e:
        print(f"Error deleting input video file: {str(e)}")

@app.route('/analyzevideo', methods=['POST'])
def analyze_video_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'})

    video_file = request.files['video']
    mtcnn = bool(int(request.form.get('mtcnn', 0)))

    if not video_file:
        return jsonify({'error': 'Invalid video file'})

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Save the video file to a temporary location
    video_path = Path('input\\' + task_id + '.mp4')
    video_file.save(video_path)

    try:
        # Submit the video analysis task and pass the task ID
        future = executor.submit(analyze_video, video_path, mtcnn, task_id)

        return jsonify({'task_id': task_id, 'message': 'Video analysis started.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/getresult/<task_id>', methods=['GET'])
def get_task_result(task_id):
    if task_id in task_results:
        result = task_results[task_id]
        if len(result) in result:
            return jsonify({'type': 'ERROR','task_id': task_id, 'message': 'Task not found or not completed yet.', 'emotions': [], 'average_emotions': [], 'traits': [], 'dominant_trait': []})
        else:
            # Calculate average emotions
            average_emotions = {}
            for entry in result:
                for emotion, value in entry.items():
                    if emotion in average_emotions:
                        average_emotions[emotion] += value
                    else:
                        average_emotions[emotion] = value

            total_entries = len(result)
            for emotion in average_emotions:
                average_emotions[emotion] /= total_entries

            # Convert scores to percentages
            for emotion, value in average_emotions.items():
                average_emotions[emotion] = round(value * 100, 2)

            # Calculate the Big Five traits scores
            openness = round((average_emotions["happy"] + average_emotions["surprise"]) / 2, 2)
            conscientiousness = round(average_emotions["neutral"], 2)
            extroversion = round(average_emotions["happy"], 2)
            agreeableness = round((average_emotions["happy"] + average_emotions["neutral"]) / 2, 2)
            neuroticism = round((average_emotions["angry"] + average_emotions["sad"] + average_emotions["fear"] +
            average_emotions["disgust"]) / 4, 2)

            # Store the results in a dictionary
            traits = {
                "openness": openness,
                "conscientiousness": conscientiousness,
                "extroversion": extroversion,
                "agreeableness": agreeableness,
                "neuroticism": neuroticism
            }

            # Determine the dominant trait
            dominant_trait = max(traits, key=traits.get)

            # Delete the output file after the result is obtained
            try:
                os.remove(f'input\\{task_id}.mp4')
            except Exception as e:
                print(f"Error deleting input result file: {str(e)}")

            return jsonify({
                'type': 'SUCCESS',
                'task_id': task_id,
                'message': 'Successfully',
                'emotions': result,
                'average_emotions': [average_emotions],
                'traits': [traits],
                'dominant_trait': [dominant_trait]
            })
    else:
        return jsonify({'type': 'ERROR','task_id': task_id, 'message': 'Task not found or not completed yet.', 'emotions': [], 'average_emotions': [], 'traits': [], 'dominant_trait': []})

if __name__ == "__main__":
    app.run()
        #app.run(host='localhost', port=8281, debug=True)

