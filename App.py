import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

a = StandardScaler()


# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)
    # Load the scaler

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

    # Ensure scaler is of type StandardScaler
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Scaler must be of type StandardScaler")
# Load normalization factors
normalization_factors_df = pd.read_csv('normalization_factors.csv')
normalization_factors = normalization_factors_df['Normalization Factor'].values

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)  # Using pose_landmarker_lite

# Initialize video capture
cap = cv2.VideoCapture(1)  # Change the index to specify which webcam to use

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
        complete_data = np.array([])
        for n in range(5):
            bundle_data = []
            
            for nn in range(33):     
             result = results_bundle[n]
             if result.pose_landmarks:
                    landmark = result.pose_landmarks.landmark[nn]
                    aaa = [0] + [x for x in range(11,17)] + [x for x in range(23,29)]
                    if nn in aaa: 
                        #print("i:", nn, "j:", n)
                        bundle_data.append(landmark.x)
                        bundle_data.append(landmark.y)
                        bundle_data.append(landmark.z)
                        if nn in [0,11,12,23,24]:
                            bundle_data.append(landmark.visibility)
            try:
                shoulder_distance = np.sqrt((bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2)
                hips_distance = np.sqrt((bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2)
                bundle_data.append(shoulder_distance)
                bundle_data.append(hips_distance)
            except Exception as e:
                pass
            complete_data = np.append(complete_data,bundle_data)  
        norm = []
        for n in range(len(complete_data)):  
            norm.append(complete_data[n] / normalization_factors[n])
        bundle = []
        bundle.append(norm)
        # Normalize the bundle data
        #normalized_bundle_data = [bundle_data[i] / normalization_factors[i] for i in range(len(bundle_data))]
        # Feed the normalized data to the xgb_model
        if len(complete_data) == 230 :
            meaning = ["['siting_down']", "['spinning']", "['standing_up']",
                        "['walking_away']", "['walking_to_camera']", 'unknown']
            
            prediction = xgb_model.predict(scaler.transform(np.array(bundle)))
            # Display the result on the screen
            cv2.putText(frame, f'Prediction: {meaning[prediction[0]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', frame)
# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
