import os
import pandas as pd
import sys


def read_files_and_calculate_median(flag_single, directory, robot_name):
    data = []
    if flag_single % 2:
        directory += 'distributed'
    else:
        directory += 'single'
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename[4:-4] == robot_name:  # Assuming the files are CSVs
                filepath = os.path.join(root, filename)
                df = pd.read_csv(filepath)

                if flag_single == 0:
                    val_ape_tran = df.iloc[-1].values[2]
                elif flag_single == 1:
                    val_ape_tran = df[df.iloc[:, 0].between(
                        3000, 3000 + 40)].iloc[0].values[2]
                elif flag_single == 2:
                    val_ape_tran = df.iloc[-1].values[3]
                elif flag_single == 3:
                    val_ape_tran = df[df.iloc[:, 0].between(
                        3000, 3000 + 40)].iloc[0].values[3]
                data.append((root, val_ape_tran))

    # Ensure there are exactly 10 files
    if len(data) < 10:
        raise ValueError("The directory must contain at least 10 CSV files.")

    if len(data) % 2 == 0:
        data.pop(0)
    # Convert list to DataFrame
    df = pd.DataFrame(data, columns=['directory', 'val_ape_tran'])

    # Calculate median
    median_value = df['val_ape_tran'].median()

    # Find all directories with the median value
    dir_with_median = df[df['val_ape_tran']
                         == median_value]['directory'].tolist()

    # If business logic only needs one directory, choose the first one
    dir_with_median = dir_with_median[0] if dir_with_median else None

    if dir_with_median is None:
        raise ValueError(
            "No directory found with the median val_ape_tran %f.", median_value)
    print(f"Directory with median val_ape_tran: {dir_with_median}")

    # Print the median value
    for file in os.listdir(dir_with_median):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir_with_median, file))
            if df.empty:
                continue
            val_ape_tran = df.iloc[-1].values[2]
            val_ape_full = df.iloc[-1].values[3]
            print(
                f"{file[4:-4]}: ape_tran: {val_ape_tran}, ape_full: {val_ape_full}")


if __name__ == "__main__":
    """
    How to run:
    python evo_median.py [flag single: s or distributed: d] [robot_name] [flag tran:0 or full:1]
    """
    directory_path = "/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test_"
    flag_single = 0
    if len(sys.argv) > 1 and sys.argv[1] == "d":
        flag_single = 1
    robot_name = "sparkal1"  # Example robot name
    if len(sys.argv) > 2 and sys.argv[2]:
        robot_name = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]:
        flag_single |= 2
    read_files_and_calculate_median(flag_single, directory_path, robot_name)
