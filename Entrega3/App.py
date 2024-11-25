import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import joblib

#Application to try out the functionality of the model

# Load the XGBoost model
xgb_model = joblib.load('Entrega3/xgb_model.pkl')

# Load the scaler
with open('Entrega3/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

    # Ensure scaler is of type StandardScaler
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Scaler must be of type StandardScaler")

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)  # Using pose_landmarker_lite

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change the index to specify which webcam to use

# Initialize a deque to store the last 5 frames
results_bundle = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = pose.process(frame_rgb)

    # Store the results in the bundle
    if len(results_bundle) == 10:
        results_bundle.popleft()
    results_bundle.append(results)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    

    if cv2.waitKey(5) & 0xFF == 27:
        break

    #Used to calculate the deltas as they are calculated in the EDA
    def calculate_apendix_speed(positions):
        speeds = []
        for i in range(1, len(positions)):
            distance = (positions[i] - positions[i-1])
            speed = distance / 1
            speeds.append(speed)
        avg_speed = sum(speeds) / len(speeds)
        return avg_speed
    #Only when the bundle is full
    if len(results_bundle) == 10:
        complete_data = np.array([])
        #Fills the complete_data array with the data of the 10 frames in the correct order
        for n in range(10):
            bundle_data = []
            for nn in range(33):
                result = results_bundle[n]
                if result.pose_landmarks:
                    landmark = result.pose_landmarks.landmark[nn]
                    aaa = [0] + [x for x in range(11,17)] + [x for x in range(23,29)]
                    if nn in aaa:
                        bundle_data.append(landmark.x)
                        bundle_data.append(landmark.y)
                        bundle_data.append(landmark.z)
                        if nn in [0,11,12,23,24]:
                            bundle_data.append(landmark.visibility)
            try:
                # The distances can only be calculated if the landmarks are present
                shoulder_distance = np.sqrt((bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2)
                hips_distance = np.sqrt((bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2)
                bundle_data.append(shoulder_distance)
                bundle_data.append(hips_distance)
            except Exception as e:
                pass
            complete_data = np.append(complete_data, bundle_data)
        #If the complete_data array is full, the deltas are calculated and the prediction is made
        if len(complete_data) == 460:
            nose_positions_y = [
                complete_data[1],
                complete_data[47],
                complete_data[93],
                complete_data[139],
                complete_data[185],
                complete_data[231],
                complete_data[277],
                complete_data[323],
                complete_data[369],
                complete_data[415]
            ]

            nose_positions_z = [
                complete_data[2],
                complete_data[48],
                complete_data[94],
                complete_data[140],
                complete_data[186],
                complete_data[232],
                complete_data[278],
                complete_data[324],
                complete_data[370],
                complete_data[416]
            ]

            nose_delta_y = calculate_apendix_speed(nose_positions_y)
            nose_delta_z = calculate_apendix_speed(nose_positions_z)

            # Example usage
            left_shoulder_positions_z = [
                complete_data[6],
                complete_data[52],
                complete_data[98],
                complete_data[144],
                complete_data[190],
                complete_data[236],
                complete_data[282],
                complete_data[328],
                complete_data[374],
                complete_data[420]
            ]

            right_shoulder_positions_z = [
                complete_data[10],
                complete_data[56],
                complete_data[102],
                complete_data[148],
                complete_data[194],
                complete_data[240],
                complete_data[286],
                complete_data[332],
                complete_data[378],
                complete_data[424]
            ]
            left_hip_positions_z = [
                complete_data[26],
                complete_data[72],
                complete_data[118],
                complete_data[164],
                complete_data[210],
                complete_data[256],
                complete_data[302],
                complete_data[348],
                complete_data[394],
                complete_data[440]
            ]

            right_hip_positions_z = [
                complete_data[30],
                complete_data[76],
                complete_data[122],
                complete_data[168],
                complete_data[214],
                complete_data[260],
                complete_data[306],
                complete_data[352],
                complete_data[398],
                complete_data[444]
            ]

            left_leg_positions_z = [
                complete_data[34],
                complete_data[80],
                complete_data[126],
                complete_data[172],
                complete_data[218],
                complete_data[264],
                complete_data[310],
                complete_data[356],
                complete_data[402],
                complete_data[448]
            ]

            right_leg_positions_z = [
                complete_data[37],
                complete_data[83],
                complete_data[129],
                complete_data[175],
                complete_data[221],
                complete_data[267],
                complete_data[313],
                complete_data[359],
                complete_data[405],
                complete_data[451]
            ]

            left_foot_positions_z = [
                complete_data[40],
                complete_data[86],
                complete_data[132],
                complete_data[178],
                complete_data[224],
                complete_data[270],
                complete_data[316],
                complete_data[362],
                complete_data[408],
                complete_data[454]
            ]

            right_foot_positions_z = [
                complete_data[43],
                complete_data[89],
                complete_data[135],
                complete_data[181],
                complete_data[227],
                complete_data[273],
                complete_data[319],
                complete_data[365],
                complete_data[411],
                complete_data[457]
            ]

            left_arm_positions_z = [
                complete_data[14],
                complete_data[60],
                complete_data[106],
                complete_data[152],
                complete_data[198],
                complete_data[244],
                complete_data[290],
                complete_data[336],
                complete_data[382],
                complete_data[428]
            ]

            right_arm_positions_z = [
                complete_data[17],
                complete_data[63],
                complete_data[109],
                complete_data[155],
                complete_data[201],
                complete_data[247],
                complete_data[293],
                complete_data[339],
                complete_data[385],
                complete_data[431]
            ]

            left_hand_positions_z = [
                complete_data[20],
                complete_data[66],
                complete_data[112],
                complete_data[158],
                complete_data[204],
                complete_data[250],
                complete_data[296],
                complete_data[342],
                complete_data[388],
                complete_data[434]
            ]

            right_hand_positions_z = [
                complete_data[23],
                complete_data[69],
                complete_data[115],
                complete_data[161],
                complete_data[207],
                complete_data[253],
                complete_data[299],
                complete_data[345],
                complete_data[391],
                complete_data[437]
            ]

            left_shoulder_delta_z = calculate_apendix_speed(left_shoulder_positions_z)
            right_shoulder_delta_z = calculate_apendix_speed(right_shoulder_positions_z)
            left_hip_delta_z = calculate_apendix_speed(left_hip_positions_z)
            right_hip_delta_z = calculate_apendix_speed(right_hip_positions_z)
            left_leg_delta_z = calculate_apendix_speed(left_leg_positions_z)
            right_leg_delta_z = calculate_apendix_speed(right_leg_positions_z)
            left_foot_delta_z = calculate_apendix_speed(left_foot_positions_z)
            right_foot_delta_z = calculate_apendix_speed(right_foot_positions_z)
            left_arm_delta_z = calculate_apendix_speed(left_arm_positions_z)
            right_arm_delta_z = calculate_apendix_speed(right_arm_positions_z)
            left_hand_delta_z = calculate_apendix_speed(left_hand_positions_z)
            right_hand_delta_z = calculate_apendix_speed(right_hand_positions_z)

            left_shoulder_positions_y = [
                complete_data[5],
                complete_data[51],
                complete_data[97],
                complete_data[143],
                complete_data[189],
                complete_data[235],
                complete_data[281],
                complete_data[327],
                complete_data[373],
                complete_data[419]
            ]

            right_shoulder_positions_y = [
                complete_data[9],
                complete_data[55],
                complete_data[101],
                complete_data[147],
                complete_data[193],
                complete_data[239],
                complete_data[285],
                complete_data[331],
                complete_data[377],
                complete_data[423]
            ]

            left_arm_positions_y = [
                complete_data[13],
                complete_data[59],
                complete_data[105],
                complete_data[151],
                complete_data[197],
                complete_data[243],
                complete_data[289],
                complete_data[335],
                complete_data[381],
                complete_data[427]
            ]

            right_arm_positions_y = [
                complete_data[16],
                complete_data[62],
                complete_data[108],
                complete_data[154],
                complete_data[200],
                complete_data[246],
                complete_data[292],
                complete_data[338],
                complete_data[384],
                complete_data[430]
            ]

            left_hand_positions_z = [
                complete_data[19],
                complete_data[65],
                complete_data[111],
                complete_data[157],
                complete_data[203],
                complete_data[249],
                complete_data[295],
                complete_data[341],
                complete_data[387],
                complete_data[433]
            ]

            right_hand_positions_z = [
                complete_data[22],
                complete_data[68],
                complete_data[114],
                complete_data[160],
                complete_data[206],
                complete_data[252],
                complete_data[298],
                complete_data[344],
                complete_data[390],
                complete_data[436]
            ]

            left_shoulder_delta_y = calculate_apendix_speed(left_shoulder_positions_y)
            right_shoulder_delta_y = calculate_apendix_speed(right_shoulder_positions_y)
            left_arm_delta_y = calculate_apendix_speed(left_arm_positions_y)
            right_arm_delta_y = calculate_apendix_speed(right_arm_positions_y)
            left_hand_delta_y = calculate_apendix_speed(left_hand_positions_z)
            right_hand_delta_y = calculate_apendix_speed(right_hand_positions_z)
            
            complete_data = np.append(complete_data, [
                
                left_shoulder_delta_z, right_shoulder_delta_z, left_hip_delta_z, right_hip_delta_z,
                left_leg_delta_z, right_leg_delta_z, left_foot_delta_z, right_foot_delta_z,
                left_arm_delta_z, right_arm_delta_z, left_hand_delta_z, right_hand_delta_z,
                left_shoulder_delta_y, right_shoulder_delta_y, left_arm_delta_y, right_arm_delta_y,
                left_hand_delta_y, right_hand_delta_y, nose_delta_y, nose_delta_z
            ])

        bundle = []
        bundle.append(complete_data)
        #If the complete_data array is full, the prediction is made
        if len(complete_data) == 480:
            #Mapping of the labels
            meaning = ["['siting_down']", "['spinning']", "['standing_up']", "['walking_to_camera']", "['walking_away']", 'unknown']
            prediction = xgb_model.predict(scaler.transform(np.array(bundle)))
            cv2.putText(frame, f'Prediction: {meaning[prediction[0]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # Display the frame 
    cv2.imshow('MediaPipe Pose', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
