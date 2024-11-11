import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)  # Using pose_landmarker_lite

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize a deque to store the last 5 frames
results_bundle = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = pose.process(frame_rgb)

    # Store the results in the bundle
    if len(results_bundle) == 5:
        results_bundle.popleft()
    results_bundle.append(results)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Process the bundle
    if len(results_bundle) == 5:
        # Create a structured dictionary for the bundle
        bundle_data = {}
        for i, result in enumerate(results_bundle):
            if result.pose_landmarks:
                for j, landmark in enumerate(result.pose_landmarks.landmark):
                    bundle_data[f'landmark_{j}_x_frame_{i+1}'] = landmark.x
                    bundle_data[f'landmark_{j}_y_frame_{i+1}'] = landmark.y
                    bundle_data[f'landmark_{j}_z_frame_{i+1}'] = landmark.z

        # Convert the bundle data to a pandas DataFrame
        
        bundle_df = pd.DataFrame([bundle_data])

        # Ensure the DataFrame columns are in the correct order
        #expected_columns = [f'landmark_{j}_{axis}_frame_{i+1}' for i in range(5) for j in range(33) for axis in ['x', 'y', 'z']]
        #bundle_df = bundle_df[expected_columns]

        # Feed the data to the xgb_model
        if(len(bundle_df.columns) == 495):

            meaning = ["['siting_down']", "['spinning']", "['standing_up']",
                        "['walking_away']", "['walking_to_camera']"]
            prediction = xgb_model.predict(bundle_df)
            print(f'Prediction: {meaning[prediction[0]]}')

            # Display the result on the screen
            cv2.putText(frame, f'Prediction: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
