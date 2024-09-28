#!/bin/sh
# Input: 
# 1: Path for log files
# # 2: Path for groundtruth files [for evo]
# # 3: time of dataset [for evo]
ROBOT_NAMES=("acl_jackal" "acl_jackal2" "sparkal1" "sparkal2" "hathor" "thoth")
# source ~/miniconda3/bin/activate env3_9

# if [ $# -gt 2 ]; then
#     DATE="$(echo "$3" | cut -c1-2)$(echo "$3" | cut -c4-5)"
# fi
while true; do
    for robot_name in "${ROBOT_NAMES[@]}"; do
        # 查找所有匹配的文件并按修改时间排序
        sorted_files=($(find $1/${robot_name}/distributed -type f -name "kimera_distributed_poses_*" -printf "%T@ %p\n" | sort -n | cut -d' ' -f2-))

        # 如果找到的文件数量大于2，进行处理
        if [ ${#sorted_files[@]} -gt 2 ]; then

            # 获取最旧和最新的文件
            min_file=${sorted_files[0]}
            max_file=${sorted_files[-1]}

            # 删除除了最旧和最新的文件之外的所有文件
            for file in "${sorted_files[@]}"; do
                if [ "$file" != "$min_file" ] && [ "$file" != "$max_file" ]; then
                    rm "$file"
                fi
            done
            # if [ $# -gt 2 ]; then
            #     echo "APE: ${robot_name}: $(evo_ape tum $2/$DATE/${robot_name}_gt_odom.csv $1/${robot_name}/${max_file} -a)"
            # fi
        fi
    done

    # 等待1分钟（60秒）
    sleep 60
done
