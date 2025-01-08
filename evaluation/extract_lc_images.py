'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-12-30 12:19:40
LastEditors: RonghaiHe && echo , && hrhkjys@qq.com
LastEditTime: 2025-01-08 12:18:00
FilePath: /src/kimera_multi/evaluation/extract_lc_images.py
Version: 1.0.0
Description: To extract images from rosbag based on Loop closure results (distance > 30m)

'''

#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import subprocess
import cv2
import time
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

DATE2DATASET = {'1207': 'campus_tunnels_12_07',
                '1014': 'campus_outdoor_10_14',
                '1208': 'campus_hybrid_12_08'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract images from rosbag based on LC results')
    parser.add_argument('--date', type=str, default='1207',
                        choices=list(DATE2DATASET.keys()),
                        help='Date of the dataset (e.g., 1207, 1014, 1208)')
    parser.add_argument('--threshold', type=float, default=30.0,
                        help='Distance threshold for LC results')
    parser.add_argument('--image_topic', type=str,
                        default='/acl_jackal/forward/color/image_raw/compressed',
                        help='Image topic name')
    parser.add_argument('--basic_bag_path', type=str,
                        default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/',
                        help='Basic path to the rosbag file')
    return parser.parse_args()


def extract_and_merge_images(bag_file1, bag_file2, timestamp1, timestamp2, robot1, robot2, gt_pose1, gt_pose2, number, distance, score, mono_inliers, stereo_inliers, output_path):
    """Extract images from two bags and merge them"""

    # Create temporary directory for output images
    output_dir = os.path.dirname(output_path) + '/temp_images'

    start_time1 = timestamp1 - 0.05
    end_time1 = timestamp1 + 0.05
    start_time2 = timestamp2 - 0.05
    end_time2 = timestamp2 + 0.05

    # Create temporary bags for both robots
    temp_bag1 = f'{number}_{robot1}_1.bag'
    temp_bag2 = f'{number}_{robot2}_2.bag'

    # Extract relevant portions from bags
    cmd1 = f'rosbag filter {bag_file1} {temp_bag1} "t.to_sec() >= {start_time1} and t.to_sec() <= {end_time1}"'
    cmd2 = f'rosbag filter {bag_file2} {temp_bag2} "t.to_sec() >= {start_time2} and t.to_sec() <= {end_time2}"'

    if not os.path.exists(temp_bag1):
        subprocess.run(cmd1, shell=True)
        subprocess.run(f'rosbag play {temp_bag1}', shell=True)
        time.sleep(5)

    img1, img2 = None, None
    img_dir1, img_dir2 = None, None

    while len(os.listdir(output_dir)) < 1:
        time.sleep(1)

    for file in os.listdir(output_dir):
        if file.startswith('frame'):
            os.rename(os.path.join(output_dir, file),
                      os.path.join(output_dir, f'{number}_1.jpg'))
            break

    if not os.path.exists(temp_bag2):
        subprocess.run(cmd2, shell=True)
        subprocess.run(f'rosbag play {temp_bag2}', shell=True)
        time.sleep(5)

    while len(os.listdir(output_dir)) < 2:
        time.sleep(1)
    for file in os.listdir(output_dir):
        if file.startswith('frame'):
            img2 = cv2.imread(os.path.join(output_dir, file))
            os.rename(os.path.join(output_dir, file),
                      os.path.join(output_dir, f'{number}_2.jpg'))

    img_dir1 = os.path.join(output_dir, f'{number}_1.jpg')
    img_dir2 = os.path.join(output_dir, f'{number}_2.jpg')
    img1 = cv2.imread(img_dir1)
    img2 = cv2.imread(img_dir2)

    # Resize images to same height
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Merge images horizontally
    merged_img = np.hstack((img1, img2))

    # Get dimensions of merged image
    height, width = merged_img.shape[:2]

    # Add padding (top, bottom, left, right)
    # Increased top and bottom padding for more text
    padding = ((80, 50), (10, 10), (0, 0))  # Increased bottom padding
    merged_img = np.pad(merged_img, padding, mode='constant')

    # Calculate text positions
    text_color = (255, 255, 255)  # White
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Distance
    cv2.putText(merged_img,
                f'Dist: {distance:.2f}',
                (20, 30),
                font, font_scale, text_color, thickness)

    # Loop closure score
    cv2.putText(merged_img,
                f'Score: {score}',
                (240, 30),
                font, font_scale, text_color, thickness)

    # Mono and stereo inliers
    cv2.putText(merged_img,
                f'Mono: {mono_inliers}',
                (460, 30),
                font, font_scale, text_color, thickness)
    cv2.putText(merged_img,
                f'Stereo: {stereo_inliers}',
                (680, 30),
                font, font_scale, text_color, thickness)

    # Groundtruth poses
    cv2.putText(merged_img,
                gt_pose1,
                (20, 60),
                font, font_scale, text_color, thickness)
    cv2.putText(merged_img,
                gt_pose2,
                (width//2 + 20, 60),
                font, font_scale, text_color, thickness)

    # Add text with adjusted positions
    # Robot names and timestamps
    cv2.putText(merged_img,
                f'{robot1} {timestamp1:.2f}',
                (20, height + 100),
                font, font_scale, text_color, thickness)
    cv2.putText(merged_img,
                f'{robot2} {timestamp2:.2f}',
                (width//2 + 20, height + 100),
                font, font_scale, text_color, thickness)

    cv2.imwrite(output_path, merged_img)

    # Cleanup
    os.remove(temp_bag1)
    os.remove(temp_bag2)
    subprocess.run(f'rm -rf {img_dir1} {img_dir2}', shell=True)


def main():
    args = parse_args()

    bag_paths = {
        robot_name: f'{args.basic_bag_path}{DATE2DATASET[args.date]}/{args.date[:2]}_{args.date[2:]}_{robot_name}.bag'
        for robot_name in ID2ROBOT
    }

    # Read the LC results CSV files
    all_high_distance_rows = []
    for robot_name in ID2ROBOT:
        inter_csv_filename = '/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/evaluation' + \
            f'/{args.date}/inter_lc_results_{args.date}_{robot_name}.csv'
        if os.path.exists(inter_csv_filename):
            try:
                df = pd.read_csv(inter_csv_filename)
            except Exception as e:
                print(f"Error reading CSV file {inter_csv_filename}: {str(e)}")
                raise
            # Convert Distance column to float and handle any non-numeric values
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
            high_distance_rows = df[df['Distance'] > args.threshold]
            all_high_distance_rows.extend(
                high_distance_rows.to_dict('records'))

        intra_csv_filename = '/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/evaluation' + \
            f'/{args.date}/intra_lc_results_{args.date}_{robot_name}.csv'
        if os.path.exists(intra_csv_filename):
            df = pd.read_csv(intra_csv_filename)
            # Convert Distance column to float and handle any non-numeric values
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
            high_distance_rows = df[df['Distance'] > args.threshold]
            if not high_distance_rows.empty:
                high_distance_rows.loc[:, 'Robot 1'] = robot_name
                high_distance_rows.loc[:, 'Robot 2'] = robot_name
            all_high_distance_rows.extend(
                high_distance_rows.to_dict('records'))

    if not all_high_distance_rows:
        print(f"No distances found above threshold {args.threshold}")
        return

    # Create output directory
    output_dir = '/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/evaluation/' + \
        f'lc_images_{args.date}'
    os.makedirs(output_dir, exist_ok=True)

    # Process each loop closure
    for row in all_high_distance_rows:
        timestamp1 = row['Timestamp 1']
        timestamp2 = row['Timestamp 2']
        number = row['Loop Closure Number']
        distance = row['Distance']
        robot1 = row['Robot 1']
        robot2 = row['Robot 2']

        gt_pose1 = f"GT Pose1: {row['GT_Pose1_X']:.2f}, {row['GT_Pose1_Y']:.2f}, {row['GT_Pose1_Z']:.2f}"
        gt_pose2 = f"GT Pose2: {row['GT_Pose2_X']:.2f}, {row['GT_Pose2_Y']:.2f}, {row['GT_Pose2_Z']:.2f}"

        score = '-'
        # if row exists 'norm_bow_score'
        if 'norm_bow_score' in row:
            score = f"{row['norm_bow_score']:.2f}"

        mono_inliers = row['mono_inliers']
        stereo_inliers = row['stereo_inliers']

        if timestamp1 > 1e12:
            timestamp1 = timestamp1 / 1e9

        if timestamp2 > 1e12:
            timestamp2 = timestamp2 / 1e9

        output_image_path = os.path.join(output_dir, 'lc_')
        if robot1 == robot2:
            output_image_path += 'intra_'
        else:
            output_image_path += 'inter_'
        output_image_path += f'{distance:.2f}_{robot1}_{robot2}_{number}.png'
        if os.path.exists(output_image_path):
            continue
        extract_and_merge_images(
            bag_paths[robot1],
            bag_paths[robot2],
            timestamp1,
            timestamp2,
            robot1,
            robot2,
            gt_pose1,
            gt_pose2,
            number,
            distance,
            score,
            mono_inliers,
            stereo_inliers,
            output_image_path
        )

    print(f'Processed {len(all_high_distance_rows)} loop closures')


if __name__ == '__main__':
    main()
