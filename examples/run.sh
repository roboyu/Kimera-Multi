#!/bin/sh
DATA_PATH_STR="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets"
NAME_TIME_STR="12_08"
LOG_DIR_STR=${DATA_PATH_STR}"/log_data_"$NAME_TIME_STR

# If no input, Run the example
if [ $# -eq 0 ]; then
    # bash run.sh
    echo "No input, run the example"
    CATKIN_WS="/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws" DATA_PATH=${DATA_PATH_STR} LOG_DIR=${LOG_DIR_STR} NAME_TIME=${NAME_TIME_STR} tmuxp load 1014-example.yaml
else
    if [ $1 = "0" ]; then
        # bash run.sh 0
        echo "Run the single example"
        CATKIN_WS="/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws" DATA_PATH=${DATA_PATH_STR} LOG_DIR=${LOG_DIR_STR} NAME_TIME=${NAME_TIME_STR} tmuxp load 1014-example_single6.yaml
    fi
fi

