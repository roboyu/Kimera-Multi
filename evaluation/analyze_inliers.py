'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-01-07 18:07:54
LastEditors: RonghaiHe && echo , && hrhkjys@qq.com
LastEditTime: 2025-01-08 12:17:40
FilePath: /src/kimera_multi/evaluation/analyze_inliers.py
Version: 1.0.0
Description: To analyze monocular inliers and stereo inliers from loop closure

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import os


def load_and_process_csv(directory):
    # Convert to absolute path if relative
    abs_directory = os.path.abspath(directory)
    file_pattern = os.path.join(abs_directory, "*_lc_results_*.csv")
    all_data = []
    for file in glob.glob(file_pattern):
        df = pd.read_csv(file)
        # Filter out "No GT data" rows
        df = df[df['Distance'] != "No GT data"]
        df['type'] = 'inter' if 'inter' in file else 'intra'
        all_data.append(df)

    if not all_data:
        raise ValueError(
            f"No CSV files found matching pattern: {file_pattern}")

    return pd.concat(all_data, ignore_index=True)


def categorize_distance(distance):
    try:
        # Convert string to float for comparison
        dist = float(distance)
        if dist <= 10:
            return '0-10m'
        elif dist <= 30:
            return '10-30m'
        else:
            return '>30m'
    except (ValueError, TypeError):
        return 'invalid'


def categorize_mono_inliers(inliers):
    try:
        val = int(inliers)
        if val == 10:
            return '10'
        elif val == 11:
            return '11'
        elif val == 12:
            return '12'
        elif val == 13:
            return '13'
        elif val == 14:
            return '14'
        elif val == 15:
            return '15'
        elif val > 15:
            return '>15'
        else:
            return '<10'
    except (ValueError, TypeError):
        return 'invalid'


def categorize_stereo_inliers(inliers):
    try:
        val = int(inliers)
        if val == 5:
            return '5'
        elif val == 6:
            return '6'
        elif val == 7:
            return '7'
        elif val == 8:
            return '8'
        elif val == 9:
            return '9'
        elif val == 10:
            return '10'
        elif val > 10:
            return '>10'
        else:
            return '<5'
    except (ValueError, TypeError):
        return 'invalid'


def format_value(x):
    """Format value to show 0 instead of empty or invalid"""
    return str(int(x)) if x is not None else '0'
    # try:
    #     val = int(x)
    #     if val == 0:
    #         return '0'  # Explicitly return '0' for zero values
    #     return str(val)
    # except (ValueError, TypeError):
    #     return '0'


def analyze_inliers(directory):
    # Load data
    data = load_and_process_csv(directory)

    # # Add categorizations
    # data['distance_category'] = data['Distance'].apply(categorize_distance)
    # data['mono_category'] = data['mono_inliers'].apply(categorize_mono_inliers)
    # data['stereo_category'] = data['stereo_inliers'].apply(
    #     categorize_stereo_inliers)

    # # Define category orders
    # mono_order = ['>15', '15', '14', '13', '12', '11', '10', '<10']
    # stereo_order = ['>10', '10', '9', '8', '7', '6', '5', '<5']
    # distance_categories = ['0-10m', '10-30m', '>30m']

    # Add categorizations
    data['distance_category'] = pd.Categorical(
        data['Distance'].apply(categorize_distance),
        categories=['0-10m', '10-30m', '>30m'],
        ordered=True
    )
    data['mono_category'] = pd.Categorical(
        data['mono_inliers'].apply(categorize_mono_inliers),
        categories=['>15', '15', '14', '13', '12', '11', '10', '<10'],
        ordered=True
    )
    data['stereo_category'] = pd.Categorical(
        data['stereo_inliers'].apply(categorize_stereo_inliers),
        categories=['>10', '10', '9', '8', '7', '6', '5', '<5'],
        ordered=True
    )

    # Split by type and create pivot tables
    inter_data = data[data['type'] == 'inter']
    intra_data = data[data['type'] == 'intra']

    # Create pivot tables with zeros for missing combinations
    processed_data = {}
    for type_name, type_data in [('inter', inter_data), ('intra', intra_data)]:
        # Create all possible combinations
        dist_cats = ['0-10m', '10-30m', '>30m']
        mono_cats = ['>15', '15', '14', '13', '12', '11', '10', '<10']
        stereo_cats = ['>10', '10', '9', '8', '7', '6', '5', '<5']

        # Create complete index for mono
        mono_idx = pd.MultiIndex.from_product([dist_cats, mono_cats],
                                              names=['distance_category', 'mono_category'])
        mono_counts = pd.Series(0, index=mono_idx).reset_index()

        # Create complete index for stereo
        stereo_idx = pd.MultiIndex.from_product([dist_cats, stereo_cats],
                                                names=['distance_category', 'stereo_category'])
        stereo_counts = pd.Series(0, index=stereo_idx).reset_index()

        # Count actual occurrences
        actual_mono = type_data.groupby(
            ['distance_category', 'mono_category'], observed=True).size().reset_index(name='count')
        actual_stereo = type_data.groupby(
            ['distance_category', 'stereo_category'], observed=True).size().reset_index(name='count')

        # Merge with complete combinations
        mono_final = mono_counts.merge(actual_mono, how='left',
                                       on=['distance_category', 'mono_category']).fillna(0)
        stereo_final = stereo_counts.merge(actual_stereo, how='left',
                                           on=['distance_category', 'stereo_category']).fillna(0)

        processed_data[type_name] = {
            'mono': mono_final,
            'stereo': stereo_final
        }

    # Create 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Plot Inter Mono Inliers
    sns.barplot(data=processed_data['inter']['mono'],
                x='distance_category',
                y='count',
                hue='mono_category',
                # hue_order=['>15', '15', '14', '13', '12', '11', '10', '<10'],
                ax=ax1)
    ax1.set_title('Inter-frame Mono Inliers Distribution')
    ax1.set_ylabel('Count')
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', padding=3)

    # Plot Intra Mono Inliers
    sns.barplot(data=processed_data['intra']['mono'],
                x='distance_category',
                y='count',
                hue='mono_category',
                # hue_order=['>15', '15', '14', '13', '12', '11', '10', '<10'],
                ax=ax2)
    ax2.set_title('Intra-frame Mono Inliers Distribution')
    ax2.set_ylabel('Count')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', padding=3)

    # Plot Inter Stereo Inliers
    sns.barplot(data=processed_data['inter']['stereo'],
                x='distance_category',
                y='count',
                hue='stereo_category',
                # hue_order=['>10', '10', '9', '8', '7', '6', '5', '<5'],
                ax=ax3)
    ax3.set_title('Inter-frame Stereo Inliers Distribution')
    ax3.set_ylabel('Count')
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%d', padding=3)

    # Plot Intra Stereo Inliers
    sns.barplot(data=processed_data['intra']['stereo'],
                x='distance_category',
                y='count',
                hue='stereo_category',
                # hue_order=['>10', '10', '9', '8', '7', '6', '5', '<5'],
                ax=ax4)
    ax4.set_title('Intra-frame Stereo Inliers Distribution')
    ax4.set_ylabel('Count')
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%d', padding=3)

    plt.tight_layout()
    plt.savefig('inliers_analysis.jpg')

    # print the data that if mono_inliners < 10
    filtered_data = data[data['mono_inliers'] < 10]
    with open('mono_less_10.csv', 'w') as f:
        filtered_data.to_csv(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze inliers from CSV files in a directory')
    parser.add_argument('--directory', type=str, default='1207',
                        help='Directory containing CSV files')
    args = parser.parse_args()
    analyze_inliers(args.directory)
