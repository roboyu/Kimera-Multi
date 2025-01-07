#usr/bin/sh

DATE="1207"

# Check if ros-noetic-image-transport-plugins is installed
if ! dpkg -l | grep ros-${ROS_DISTRO}-image-transport-plugins; then
    sudo apt-get install -y ros-${ROS_DISTRO}-image-transport-plugins
fi

source /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/devel/setup.bash

# Extract sub-rosbag for specific image and merge them into one image
gnome-terminal --title="run" -- /bin/bash -c "source ~/miniconda3/bin/activate env3_9; python extract_lc_images.py --date $DATE"

# Run launch file to extract image from the topic
gnome-terminal --title="run" -- /bin/bash -c "roslaunch kimera_multi extract_lc_images.launch"

mkdir -p "./lc_images_$DATE"

while true; do
    sleep 1
    # Check for jpg files and move them safely
    if [ $(find "$HOME/.ros" -maxdepth 1 -name "*.jpg" | wc -l) -gt 0 ] ; then
        find "$HOME/.ros" -name "*.jpg" -exec mv {} "./lc_images_$DATE/temp_images/" \;
    fi
done