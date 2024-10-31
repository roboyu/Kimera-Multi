#!/bin/bash

# Define the target script and the file to check
TARGET_SCRIPT="run.sh"
CHECK_PYTHON_SCRIPT="evo_real_time.py"
DIR_ORIGIN="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test"
DIR_DEST="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test_distributed_comm_50/"

# Loop to run the target script 10 times
for i in {5..10}
do
    echo "Running the target script for the $i-th time"
    # Open another terminal named "run" and Run the target script
    gnome-terminal --title="run" -- /bin/bash -c "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples; bash $TARGET_SCRIPT"
    # bash $TARGET_SCRIPT
    sleep 5
    # Loop to check if the Python script is running
    while true; do
        if [ $(pgrep -f "$CHECK_PYTHON_SCRIPT" | wc -l) -lt 2 ]; then
            # Run some commands if the Python script is running
            echo "Try to kill the rosnode"
            rosnode kill -a
            sleep 5
            echo "Success. Now try to kill tmux session"
            tmux kill-server
            sleep 5
            echo "Success. Now try to modify the test directory"
            mv $DIR_ORIGIN $DIR_ORIGIN"$i"
            mv $DIR_ORIGIN"$i" $DIR_DEST
            echo "Success. Now try to restart the experiment"
            break
        else
            echo "Python script is running, checking again..."
            sleep 600
        fi
    done
done

echo "Done. Go and check the results in ${DIR_DEST}!"