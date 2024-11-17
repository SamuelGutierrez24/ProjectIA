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
cap = cv2.VideoCapture(0)  # Change the index to specify which webcam to use

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

    def calculate_apendix_speed(positions):
        speeds = []
        for i in range(1, len(positions)):
            # Calculate the Euclidean distance between consecutive positions
            distance = np.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            # Assuming the time interval between each position is 1 unit
            speed = distance / 1  # Replace 1 with the actual time interval if different
            speeds.append(speed)
        avg_speed = sum(speeds) / len(speeds)
        return avg_speed

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
        if(len(complete_data) == 230):
            right_leg_positions = [
                [complete_data[41], complete_data[42], complete_data[43]],
                [complete_data[87], complete_data[88], complete_data[89]],
                [complete_data[133], complete_data[134], complete_data[135]],
                [complete_data[180], complete_data[181], complete_data[183]],
                [complete_data[226], complete_data[227], complete_data[228]]
            ]

            left_leg_positions = [
                [complete_data[38], complete_data[39], complete_data[40]],
                [complete_data[84], complete_data[85], complete_data[86]],
                [complete_data[130], complete_data[131], complete_data[132]],
                [complete_data[176], complete_data[177], complete_data[178]],
                [complete_data[222], complete_data[223], complete_data[224]]
            ]

            left_arm_positions = [
                [complete_data[18], complete_data[19], complete_data[20]],
                [complete_data[64], complete_data[65], complete_data[66]],
                [complete_data[110], complete_data[111], complete_data[112]],
                [complete_data[156], complete_data[157], complete_data[158]],
                [complete_data[202], complete_data[203], complete_data[204]]
            ]

            right_arm_positions = [
                [complete_data[21], complete_data[22], complete_data[23]],
                [complete_data[67], complete_data[68], complete_data[69]],
                [complete_data[113], complete_data[114], complete_data[115]],
                [complete_data[159], complete_data[160], complete_data[161]],
                [complete_data[205], complete_data[206], complete_data[207]]
            ]
            left_leg_speed = calculate_apendix_speed(left_leg_positions)
            right_leg_speed = calculate_apendix_speed(right_leg_positions)
            left_arm_speed = calculate_apendix_speed(left_arm_positions)
            right_arm_speed = calculate_apendix_speed(right_arm_positions)
            complete_data = np.append(complete_data, [left_leg_speed, right_leg_speed, left_arm_speed, right_arm_speed])
        #norm = []
        #for n in range(len(complete_data)):  
            #norm.append(complete_data[n] / normalization_factors[n])
        bundle = []
        bundle.append(complete_data)
        # Normalize the bundle data
        #normalized_bundle_data = [bundle_data[i] / normalization_factors[i] for i in range(len(bundle_data))]
        # Feed the normalized data to the xgb_model
        if len(complete_data) > 0 :
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
