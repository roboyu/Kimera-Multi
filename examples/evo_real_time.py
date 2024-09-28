import signal
import os
import glob
import copy
import pandas as pd
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
import matplotlib.pyplot as plt

# 设置目录和前缀
LOG_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/log_data_12_08/'
GT_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1208/'
APE_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test/'
PREFIX = 'kimera_distributed_poses_tum_'
ROBOT_NUM = 6
ROBOT_NAMES = ['acl_jackal', 'acl_jackal2', 'sparkal1', 'sparkal2', 'hathor', 'thoth']
traj_ref = [None] * ROBOT_NUM

# 关闭所有现有的图形窗口
plt.close('all')

# 创建一个子图
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle('APE Translation')

ape_dict = {}
for num in range(ROBOT_NUM):
    ape_dict[ROBOT_NAMES[num]] = pd.DataFrame(columns=['ts', 'count', 'trans', 'full'])
          

# 读取GT轨迹
for num in range(ROBOT_NUM):
    ref_file = os.path.join(GT_DIR, f'modified_{ROBOT_NAMES[num]}_gt_odom.tum')
    traj_ref[num] = file_interface.read_tum_trajectory_file(ref_file)


# 设置evo计算ape
ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
ape_full = metrics.APE(metrics.PoseRelation.full_transformation)
max_diff = 0.01

# 定义信号处理函数
def signal_handler(sig, frame):
    for num in range(ROBOT_NUM):
        ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
        print(f'Saved APE data for {ROBOT_NAMES[num]}')

# 捕获 SIGTERM 和 SIGHUP 信号
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

process_old = False
# oldest_file = None

def main():
    try:
        while True:
            for num in range(ROBOT_NUM):
                LOG_DIR_ROBOT = os.path.join(LOG_DIR, ROBOT_NAMES[num], 'distributed/')
                # 存储文件列表的字典，键是前缀
                files = []

                # 扫描目录
                for file_path in glob.glob(os.path.join(LOG_DIR_ROBOT, f'{PREFIX}*.tum')):
                    # print(file_path)
                    # file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                    # if file_prefix not in files_dict:
                    #     files_dict[file_prefix] = []
                    files.append(file_path)
                
                if len(files) < 1: 
                    # print(f'Not enough files for {ROBOT_NAMES[num]}')
                    continue
                newest_file = None
                
                # 按最后修改时间排序
                files.sort(key=lambda x: os.path.getmtime(x))

                # 保留最新的和最旧的文件
                newest_file = files[-1]
                # if not process_old:
                #     oldest_file = files[0]

                # 删除其他文件
                for file in files[1:-1]:
                    # print(f'Removing {file}')
                    os.remove(file)
                
                # 获取文件名称的数字部分
                newest_file_num = int(newest_file.split('_')[-1].split('.')[0])
                print(f'Processing {newest_file_num} of {ROBOT_NAMES[num]}')
                
                traj_est = file_interface.read_tum_trajectory_file(newest_file)
                traj_ref_, traj_est = sync.associate_trajectories(traj_ref[num], traj_est, max_diff)
                
                traj_est_aligned = copy.deepcopy(traj_est)
                traj_est_aligned.align(traj_ref_, correct_scale=False, correct_only_scale=False)

                # 计算APE
                data = (traj_ref_, traj_est_aligned)
                ape_trans.process_data(data)
                ape_full.process_data(data)

                new_row = pd.DataFrame([{'ts': newest_file_num, 'count': len(traj_est_aligned.timestamps), 'trans': ape_trans.get_statistic(metrics.StatisticsType.rmse), 'full': ape_full.get_statistic(metrics.StatisticsType.rmse)}])
                ape_dict[ROBOT_NAMES[num]] = pd.concat([ape_dict[ROBOT_NAMES[num]], new_row], ignore_index=True)

            # Draw the APE
            ax2 = [None] * ROBOT_NUM
            for num in range(ROBOT_NUM):
                ax[int(num/3), int(num%3)].plot(ape_dict[ROBOT_NAMES[num]]['ts'], ape_dict[ROBOT_NAMES[num]]['trans'], color='blue', label='APE')
                # print(f"{ape_dict[ROBOT_NAMES[num]]['ts']}, {ROBOT_NAMES[num]}")
                ax2[num] = ax[int(num/3), int(num%3)].twinx()
                ax2[num].plot(ape_dict[ROBOT_NAMES[num]]['ts'], ape_dict[ROBOT_NAMES[num]]['count'], color='red', label='num')

                # if(len(ape_dict[ROBOT_NAMES[num]]['ts']) < 2):
                ax[int(num/3), int(num%3)].set_title(f'{ROBOT_NAMES[num]}')
                ax[int(num/3), int(num%3)].set_xlabel('Timestamp')
                if not num % 3:
                    if not num:
                        ax[0,0].legend()
                    else:
                        ax2[num].legend()
                    ax[int(num/3), 0].set_ylabel('APE Translation')
                if num % 3 == 2:
                    ax2[num].set_ylabel('number of poses')
            
            plt.draw()
            plt.pause(5)
            plt.savefig(os.path.join(APE_DIR, 'ape.jpg'))
            for num in range(ROBOT_NUM):
                ax2[num].clear()
                ax2[num].set_yticks([])
                ax[int(num/3), int(num%3)].clear()
                # ax2[num].line = []
                # ax[int(num/3), int(num%3)].line = []
                
            print('-'*10)

    except Exception as e:
        print(f'Exiting for {e}')
        plt.close('all')
        for num in range(ROBOT_NUM):
            ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
            print(f'Saved APE data for {ROBOT_NAMES[num]}')

if __name__ == '__main__':
    main()