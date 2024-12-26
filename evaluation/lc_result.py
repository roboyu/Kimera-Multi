import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np

ID2ROBOT = [
    'acl_jackal',
    'acl_jackal2',
    'sparkal1',
    'sparkal2',
    'hathor',
    'thoth',
    'apis',
    'sobek'
]


def read_groundtruth_tum(file_path):
    # Read the ground truth poses from TUM file
    groundtruth_data = pd.read_csv(file_path, sep=' ', header=None)
    groundtruth_data.columns = ['timestamp',
                                'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return groundtruth_data


def find_closest_pose(timestamp, groundtruth_data, tolerance=0.01):
    # Find the closest pose within the given tolerance
    closest_pose = None
    min_diff = float('inf')
    for _, row in groundtruth_data.iterrows():
        diff = abs(row['timestamp'] - timestamp)
        if diff < min_diff and diff <= tolerance:
            min_diff = diff
            closest_pose = row[1:].values
    return closest_pose


def calculate_relative_pose(pose1, pose2):
    # Extract quaternion and translation from poses
    q1 = pose1[3:]
    t1 = pose1[:3]
    q2 = pose2[3:]
    t2 = pose2[:3]

    # Convert quaternions to rotation matrices
    R1 = R.from_quat(q1).as_matrix()
    R2 = R.from_quat(q2).as_matrix()

    # Calculate relative rotation
    R_rel = R2 @ R1.T

    # Convert relative rotation matrix back to quaternion
    q_rel = R.from_matrix(R_rel).as_quat()

    # Calculate relative translation
    t_rel = t2 - R_rel @ t1

    # Calculate distance
    distance = np.linalg.norm(t_rel)

    # Calculate rotation angle
    rotation = R.from_quat(q_rel)
    angle = rotation.magnitude()

    return q_rel, t_rel, distance, angle


def main(loop_closure_file_prefix, keyframes_files_prefix, groundtruth_files_prefix, num_robots, output_file):
    # Read the loop closure data
    loop_closure_files = [
        f"{loop_closure_file_prefix}/{ID2ROBOT[i]}/distributed/loop_closures.csv" for i in range(num_robots)
    ]
    loop_closure_data = {i: pd.read_csv(
        file) for i, file in enumerate(loop_closure_files)}

    # Read the ground truth data for all robots
    groundtruth_files = [
        f"{groundtruth_files_prefix}modified_{ID2ROBOT[i]}_gt_odom.tum" for i in range(num_robots)
    ]
    groundtruth_data = {i: read_groundtruth_tum(
        file) for i, file in enumerate(groundtruth_files)}

    # Read the keyframes data for all robots, use keyframe's ID mapping timestamp
    keyframes_files = [
        f"{keyframes_files_prefix}{ID2ROBOT[i]}/distributed/kimera_distributed_keyframes.csv" for i in range(num_robots)
    ]
    keyframes_data = {i: pd.read_csv(
        file) for i, file in enumerate(keyframes_files)}

    # Create a dictionary to store the keyframe timestamps by ID for each robot
    keyframes_dict = {
        i: {row['keyframe_id']: row['keyframe_stamp_ns'] /
            1e9 for _, row in keyframes_data[i].iterrows()}
        for i in range(num_robots)
    }

    earliest_timestamp = [keyframes_dict[i].get(0) for i in range(num_robots)]

    for i in range(num_robots):
        with open(output_file + ID2ROBOT[i] + '.csv', 'w') as f:
            # Write the CSV header
            f.write("Loop Closure Number,Robot 1,Relative Time 1,Robot 2,Relative Time 2,Distance,Rotation Angle (radians),Estimated Distance, Estimated Angle(Radian),Timestamp 1,Timestamp 2,Relative Rotation Quaternion,Relative Translation Vector,Estimated Relative Rotation, Estimated Relative Translation\n")

            # Iterate through loop closure data to calculate relative poses
            for index, row in loop_closure_data[i].iterrows():
                robot1 = row['robot1']
                robot2 = row['robot2']
                keyframe_id1 = row['pose1']
                keyframe_id2 = row['pose2']

                timestamp1 = keyframes_dict[robot1].get(keyframe_id1)
                timestamp2 = keyframes_dict[robot2].get(keyframe_id2)

                estimated_relative_R = np.array(
                    [row['qw'], row['qx'], row['qy'], row['qz']])
                estimated_relative_t = row[8:11].values
                estimated_distance = np.linalg.norm(estimated_relative_t)
                estimated_angle = R.from_quat(estimated_relative_R).magnitude()

                if timestamp1 is not None and timestamp2 is not None:
                    pose1 = find_closest_pose(
                        timestamp1, groundtruth_data[robot1], tolerance=0.1)
                    pose2 = find_closest_pose(
                        timestamp2, groundtruth_data[robot2], tolerance=0.1)

                    if pose1 is not None and pose2 is not None:
                        relative_time1 = timestamp1 - \
                            earliest_timestamp[int(robot1)]
                        relative_time2 = timestamp2 - \
                            earliest_timestamp[int(robot2)]

                        # Calculate the relative pose, distance, and rotation angle
                        q_rel, t_rel, distance, angle = calculate_relative_pose(
                            pose1, pose2)

                        f.write(
                            f"{index},{ID2ROBOT[int(robot1)]},{relative_time1},{ID2ROBOT[int(robot2)]},{relative_time2},{distance},{angle},{estimated_distance},{estimated_angle},{timestamp1},{timestamp2},{q_rel},{t_rel},{estimated_relative_R},{estimated_relative_t}\n")
                    else:
                        f.write(
                            f"{index},{ID2ROBOT[int(robot1)]},,{ID2ROBOT[int(robot2)]},,No GT data,,{estimated_distance},{estimated_angle},{timestamp1},{timestamp2}\n")
                else:
                    f.write(
                        f"{index},{ID2ROBOT[int(robot1)]},,{ID2ROBOT[int(robot2)]},,No keyframe data,,{estimated_distance},{estimated_angle},{timestamp1},{timestamp2}\n")


if __name__ == "__main__":
    # Path to your loop closure CSV file
    loop_closure_file_prefix = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/campus_tunnels_12_07/log_data_12_07/'
    # Prefix for your keyframes CSV files
    keyframes_files_prefix = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/campus_tunnels_12_07/log_data_12_07/'
    # Prefix for your ground truth TUM files
    groundtruth_files_prefix = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1207/'
    # Number of robots
    num_robots = 6
    # Output file for results
    output_file = 'lc_results_1207_'
    main(loop_closure_file_prefix, keyframes_files_prefix,
         groundtruth_files_prefix, num_robots, output_file)
