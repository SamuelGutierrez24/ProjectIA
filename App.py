import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import joblib

a = StandardScaler()


# Load the XGBoost model
xgb_model = joblib.load('xgb_model.pkl')
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
            distance = (positions[i] - positions[i-1])
            speed = distance / 1
            speeds.append(speed)
        avg_speed = sum(speeds) / len(speeds)
        return avg_speed

    if len(results_bundle) == 5:
        complete_data = np.array([])
        for n in range(5):
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
                shoulder_distance = np.sqrt((bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2 + (bundle_data[4] - bundle_data[8]) ** 2)
                hips_distance = np.sqrt((bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2 + (bundle_data[24] - bundle_data[28]) ** 2)
                bundle_data.append(shoulder_distance)
                bundle_data.append(hips_distance)
            except Exception as e:
                pass
            complete_data = np.append(complete_data, bundle_data)
        if len(complete_data) == 230:
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            nose_positions_y = [
                complete_data[1],
                complete_data[47],
                complete_data[93],
                complete_data[139],
                complete_data[185]
            ]

            nose_positions_z = [
                complete_data[2],
                complete_data[48],
                complete_data[94],
                complete_data[140],
                complete_data[186]
            ]

            nose_delta_y = calculate_apendix_speed(nose_positions_y)
            nose_delta_z = calculate_apendix_speed(nose_positions_z)

            # Example usage
            left_shoulder_positions_z = [
                complete_data[6],
                complete_data[52],
                complete_data[98],
                complete_data[144],
                complete_data[190]
            ]

            right_shoulder_positions_z = [
                complete_data[10],
                complete_data[56],
                complete_data[102],
                complete_data[148],
                complete_data[194]
                ]
            left_hip_positions_z = [
                complete_data[26],
                complete_data[72],
                complete_data[118],
                complete_data[164],
                complete_data[210]
            ]

            right_hip_positions_z = [
                complete_data[30],
                complete_data[76],
                complete_data[122],
                complete_data[168],
                complete_data[214]
            ]

            left_leg_positions_z = [
                complete_data[34],
                complete_data[80],
                complete_data[126],
                complete_data[172],
                complete_data[218]
            ]

            right_leg_positions_z = [
                complete_data[37],
                complete_data[83],
                complete_data[129],
                complete_data[175],
                complete_data[221]
            ]

            left_foot_positions_z = [
                complete_data[40],
                complete_data[86],
                complete_data[132],
                complete_data[178],
                complete_data[224]
            ]

            right_foot_positions_z = [
                complete_data[43],
                complete_data[89],
                complete_data[135],
                complete_data[181],
                complete_data[227]
            ]

            left_arm_positions_z = [
                complete_data[14],
                complete_data[60],
                complete_data[106],
                complete_data[152],
                complete_data[198]
            ]

            right_arm_positions_z = [
                complete_data[17],
                complete_data[63],
                complete_data[109],
                complete_data[155],
                complete_data[201]
                ]

            left_hand_positions_z = [
                complete_data[20],
                complete_data[66],
                complete_data[112],
                complete_data[158],
                complete_data[204]
            ]

            right_hand_positions_z = [
                complete_data[23],
                complete_data[69],
                complete_data[115],
                complete_data[161],
                complete_data[207]
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
                complete_data[189]
            ]

            right_shoulder_positions_y = [
                complete_data[9],
                complete_data[55],
                complete_data[101],
                complete_data[147],
                complete_data[193]
                ]

            left_arm_positions_y = [
                complete_data[13],
                complete_data[59],
                complete_data[105],
                complete_data[151],
                complete_data[197]
            ]

            right_arm_positions_y = [
                complete_data[16],
                complete_data[62],
                complete_data[108],
                complete_data[154],
                complete_data[200]
                ]

            left_hand_positions_z = [
                complete_data[19],
                complete_data[65],
                complete_data[111],
                complete_data[157],
                complete_data[203]
            ]

            right_hand_positions_z = [
                complete_data[22],
                complete_data[68],
                complete_data[114],
                complete_data[160],
                complete_data[206]
                ] 

            left_shoulder_delta_y = calculate_apendix_speed(left_shoulder_positions_y)
            right_shoulder_delta_y = calculate_apendix_speed(right_shoulder_positions_y)
            left_arm_delta_y = calculate_apendix_speed(left_arm_positions_y)
            right_arm_delta_y = calculate_apendix_speed(right_arm_positions_y)
            left_hand_delta_y = calculate_apendix_speed(left_hand_positions_z)
            right_hand_delta_y = calculate_apendix_speed(right_hand_positions_z)
            left_shoulder_positions_x = [
                complete_data[4],
                complete_data[50],
                complete_data[96],
                complete_data[142],
                complete_data[188]
            ]
            right_shoulder_positions_x = [
                complete_data[8],
                complete_data[54],
                complete_data[100],
                complete_data[146],
                complete_data[192]
                ]
            left_hip_positions_x = [
                complete_data[24],
                complete_data[70],
                complete_data[116],
                complete_data[162],
                complete_data[208]
            ]

            right_hip_positions_x = [
                complete_data[28],
                complete_data[74],
                complete_data[120],
                complete_data[166],
                complete_data[212]
            ]


            left_leg_positions_x = [
                complete_data[32],
                complete_data[78],
                complete_data[124],
                complete_data[170],
                complete_data[216]
            ]

            right_leg_positions_x = [
                complete_data[35],
                complete_data[81],
                complete_data[127],
                complete_data[173],
                complete_data[219]
            ]

            left_foot_positions_x = [
                complete_data[38],
                complete_data[84],
                complete_data[130],
                complete_data[176],
                complete_data[222]
            ]

            right_foot_positions_x = [
                complete_data[41],
                complete_data[87],
                complete_data[133],
                complete_data[179],
                complete_data[225]
            ]

            left_arm_positions_x = [
                complete_data[12],
                complete_data[58],
                complete_data[104],
                complete_data[150],
                complete_data[196]
            ]

            right_arm_positions_x = [
                complete_data[15],
                complete_data[61],
                complete_data[107],
                complete_data[153],
                complete_data[199]
                ]

            left_hand_positions_x = [
                complete_data[18],
                complete_data[64],
                complete_data[110],
                complete_data[156],
                complete_data[202]
            ]

            right_hand_positions_x = [
                complete_data[21],
                complete_data[67],
                complete_data[113],
                complete_data[159],
                complete_data[205]
                ] 

            left_shoulder_delta_x = calculate_apendix_speed(left_shoulder_positions_x)
            right_shoulder_delta_x = calculate_apendix_speed(right_shoulder_positions_x)
            left_hip_delta_x = calculate_apendix_speed(left_hip_positions_x)
            right_hip_delta_x = calculate_apendix_speed(right_hip_positions_x)
            left_leg_delta_x = calculate_apendix_speed(left_leg_positions_x)
            right_leg_delta_x = calculate_apendix_speed(right_leg_positions_x)
            left_foot_delta_x = calculate_apendix_speed(left_foot_positions_x)
            right_foot_delta_x = calculate_apendix_speed(right_foot_positions_x)
            left_arm_delta_x = calculate_apendix_speed(left_arm_positions_x)
            right_arm_delta_x = calculate_apendix_speed(right_arm_positions_x)
            left_hand_delta_x = calculate_apendix_speed(left_hand_positions_x)
            right_hand_delta_x = calculate_apendix_speed(right_hand_positions_x)

            complete_data = np.append(complete_data, [
                left_shoulder_delta_x, right_shoulder_delta_x, left_hip_delta_x, right_hip_delta_x,
                left_leg_delta_x, right_leg_delta_x, left_foot_delta_x, right_foot_delta_x,
                left_arm_delta_x, right_arm_delta_x, left_hand_delta_x, right_hand_delta_x,
                left_shoulder_delta_z, right_shoulder_delta_z, left_hip_delta_z, right_hip_delta_z,
                left_leg_delta_z, right_leg_delta_z, left_foot_delta_z, right_foot_delta_z,
                left_arm_delta_z, right_arm_delta_z, left_hand_delta_z, right_hand_delta_z,
                left_shoulder_delta_y, right_shoulder_delta_y, left_arm_delta_y, right_arm_delta_y,
                left_hand_delta_y, right_hand_delta_y, nose_delta_y, nose_delta_z
            ])

        bundle = []
        bundle.append(complete_data)
        if len(complete_data) ==262:
            meaning = ["['siting_down']", "['spinning']", "['standing_up']", "['walking_away']", "['walking_to_camera']", 'unknown']
            prediction = xgb_model.predict(scaler.transform(np.array(bundle)))
            cv2.putText(frame, f'Prediction: {meaning[prediction[0]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
