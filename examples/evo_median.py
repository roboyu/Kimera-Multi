import os
import pandas as pd
import argparse


def read_files_and_calculate_median(args, directory):
    data = []
    if args.flag_single == 1:
        directory += 'distributed'
    elif args.flag_single == 2:
        directory += 'distributed_comm_50'
    else:
        directory += 'single'
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename[4:-4] == args.robot:  # Assuming the files are CSVs
                filepath = os.path.join(root, filename)
                df = pd.read_csv(filepath)

                if args.flag_single == 0:
                    val_ape_tran = df.iloc[-1].values[2]
                elif args.flag_single == 1:
                    val_ape_tran = df[df.iloc[:, 0].between(
                        args.value, args.value + 40)].iloc[0].values[2]
                elif args.flag_single == 2:
                    val_ape_tran = df.iloc[-1].values[3]
                elif args.flag_single == 3:
                    val_ape_tran = df[df.iloc[:, 0].between(
                        args.value, args.value + 40)].iloc[0].values[3]
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
            if args.flag_single == 0:
                val_ape_tran = df.iloc[-1].values[2]
                val_ape_full = df.iloc[-1].values[3]
            else:
                val_ape_tran = df[df.iloc[:, 0].between(
                    args.value, args.value + 40)].iloc[0].values[2]
                val_ape_full = df[df.iloc[:, 0].between(
                    args.value, args.value + 40)].iloc[0].values[3]

                if (df.iloc[:, 0] == 2000).any():
                    print(
                        f"{file[4:-4]} in 1: ape_tran: {df[df.iloc[:, 0] >= 2000].iloc[:, 2].median()}, "
                        f"ape_full: {df[df.iloc[:, 0] >= 2000].iloc[:, 3].median()}")
                else:
                    print(
                        f"{file[4:-4]} in 2: ape_tran: {df.iloc[:, 2].median()}, "
                        f"ape_full: {df.iloc[:, 3].median()}")
            print(
                f"{file[4:-4]}: ape_tran: {val_ape_tran}, ape_full: {val_ape_full}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate median val_ape_tran from CSV files.")
    parser.add_argument("--flag_single", type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
                        help="Flag single: 0: single, 1: distributed, 2: distributed with comm 50, or 3,4,5: corresponding to pick ape_full")
    parser.add_argument("--robot", default='sparkal1',
                        help="Name of the robot")
    parser.add_argument("--value", type=int, default=2000,
                        help="Value to use for selection")
    parser.add_argument("--value_end", type=int, default=10000,
                        help="Value to use for selection as stability")
    args = parser.parse_args()

    directory_path = "/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test_"

    read_files_and_calculate_median(args, directory_path)
