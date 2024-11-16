import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Load normalization factors
normalization_factors_df = pd.read_csv('normalization_factors.csv')
normalization_factors = normalization_factors_df['Normalization Factor'].values

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
    

    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Process the bundle
    if len(results_bundle) == 5:
        # Create a structured dictionary for the bundle
        bundle_data = []
        for n in range(33):
            for nn in range(5):
             result = results_bundle[nn]
             if result.pose_landmarks:
                    landmark = result.pose_landmarks.landmark[n]
                    if n not in range(1, 11): 
                        #print("i:", nn, "j:", n)
                        bundle_data.append(landmark.x)
                        bundle_data.append(landmark.y)
                        bundle_data.append(landmark.z)

        # Normalize the bundle data
        normalized_bundle_data = [bundle_data[i] / normalization_factors[i] for i in range(len(bundle_data))]
        print(normalized_bundle_data)
        # Feed the normalized data to the xgb_model
        if len(normalized_bundle_data) == 345:
            scaler = StandardScaler()
            meaning = ["['siting_down']", "['spinning']", "['standing_up']",
                        "['walking_away']", "['walking_to_camera']", 'unknown']
            
            prediction = xgb_model.predict(scaler.fit_transform(np.array([normalized_bundle_data])))
            # Display the result on the screen
            cv2.putText(frame, f'Prediction: {meaning[prediction[0]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', frame)
# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
