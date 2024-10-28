import matplotlib.pyplot as plt
import signal
import os
import glob
import copy
import sys
import time
import pandas as pd
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
plot.apply_settings(SETTINGS)

# 设置目录和前缀
LOG_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/log_data_12_08/'
GT_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1208/'
APE_DIR = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/test/'
PREFIX = 'kimera_distributed_poses_tum_'
INTERVAL = 5  # 间隔计算evo的时间
ROBOT_NUM = 6
ROBOT_NAMES = ['acl_jackal', 'acl_jackal2',
               'sparkal1', 'sparkal2', 'hathor', 'thoth']
traj_ref = [None] * ROBOT_NUM

if not os.path.exists(APE_DIR):
    os.makedirs(APE_DIR)

ape_dict = {}
for num in range(ROBOT_NUM):
    ape_dict[ROBOT_NAMES[num]] = pd.DataFrame(
        columns=['ts', 'count', 'trans', 'full'])


# 读取GT轨迹
for num in range(ROBOT_NUM):
    ref_file = os.path.join(GT_DIR, f'modified_{ROBOT_NAMES[num]}_gt_odom.tum')
    traj_ref[num] = file_interface.read_tum_trajectory_file(ref_file)


# 设置evo计算ape
ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
ape_full = metrics.APE(metrics.PoseRelation.full_transformation)
max_diff = 0.01

# 定义信号处理函数


def signal_handler(_sig, _frame):
    plt.close('all')
    if not os.path.exists(APE_DIR):
        os.makedirs(APE_DIR)
    for num in range(ROBOT_NUM):
        ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
            APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
        print(f'Saved APE data for {ROBOT_NAMES[num]}')
    sys.exit(0)


# 捕获 SIGTERM 和 SIGHUP 信号
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

newest_file_num = 0


def main(flag_single=0, retry_count=10):
    global newest_file_num

    TYPE_DIR = 'distributed/'
    newest_file = None
    if flag_single:
        TYPE_DIR = 'single/'
        # newest_file_num = 0

    attempt = 0
    while attempt < retry_count:

        try:
            while True:
                start_time = time.time()
                plt.close('all')
                ad_traj_by_label = [{} for _ in range(ROBOT_NUM)]
                for num in range(ROBOT_NUM):
                    LOG_DIR_ROBOT = os.path.join(
                        LOG_DIR, ROBOT_NAMES[num], TYPE_DIR)

                    if flag_single:
                        if (num == 0):
                            newest_file_num += INTERVAL
                        # if (newest_file_num < 30):
                        #     time.sleep(INTERVAL)
                        #     break
                        newest_file = os.path.join(
                            LOG_DIR_ROBOT, 'traj_pgo.tum')

                        # 如果文件不存在或文件是空的，跳过
                        if not os.path.exists(newest_file) or not os.path.getsize(newest_file):
                            continue

                        # 获取文件内容行数
                        with open(newest_file) as f:
                            row_num = sum(1 for _ in f)
                        if row_num < 100:
                            continue

                    else:
                        # 存储所有位姿文件
                        files = []

                        # 扫描目录
                        for file_path in glob.glob(os.path.join(LOG_DIR_ROBOT, f'{PREFIX}*.tum')):
                            files.append(file_path)

                        if len(files) < 1:
                            # print(f'Not enough files for {ROBOT_NAMES[num]}')
                            continue
                        newest_file = None

                        # 按最后修改时间排序
                        files.sort(key=lambda x: os.path.getmtime(x))

                        # 保留最新的和最旧的文件
                        newest_file = files[-1]

                        # 删除其他文件
                        for file in files[1:-1]:
                            # print(f'Removing {file}')
                            os.remove(file)

                        # 获取文件名称的数字部分
                        newest_file_num = int(
                            newest_file.split('_')[-1].split('.')[0])

                    print(
                        f'Processing {newest_file_num} of {ROBOT_NAMES[num]}')
                    # 获取文件内容行数
                    if flag_single:
                        print(
                            f'File of {ROBOT_NAMES[num]} has {row_num} lines')

                    traj_est = file_interface.read_tum_trajectory_file(
                        newest_file)
                    traj_ref_, traj_est = sync.associate_trajectories(
                        traj_ref[num], traj_est, max_diff)

                    traj_est_aligned = copy.deepcopy(traj_est)
                    traj_est_aligned.align(
                        traj_ref_, correct_scale=False, correct_only_scale=False)

                    ad_traj_by_label[num] = {
                        "est": traj_est_aligned, "ref": traj_ref_}

                    # 计算APE
                    data = (traj_ref_, traj_est_aligned)
                    ape_trans.process_data(data)
                    ape_full.process_data(data)

                    new_row = pd.DataFrame([{'ts': newest_file_num, 'count': len(traj_est_aligned.timestamps), 'trans': ape_trans.get_statistic(
                        metrics.StatisticsType.rmse), 'full': ape_full.get_statistic(metrics.StatisticsType.rmse)}])
                    ape_dict[ROBOT_NAMES[num]] = pd.concat(
                        [ape_dict[ROBOT_NAMES[num]], new_row], ignore_index=True)

                # Draw the APE
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                ax2 = [None] * ROBOT_NUM
                for num in range(ROBOT_NUM):
                    ax[int(num/3), int(num % 3)].plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                                                      ape_dict[ROBOT_NAMES[num]]['trans'], color='blue', label='APE')
                    # print(f"{ape_dict[ROBOT_NAMES[num]]['ts']}, {ROBOT_NAMES[num]}")
                    ax2[num] = ax[int(num/3), int(num % 3)].twinx()
                    ax2[num].plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                                  ape_dict[ROBOT_NAMES[num]]['count'], color='red', label='num')

                    # if(len(ape_dict[ROBOT_NAMES[num]]['ts']) < 2):
                    ax[int(num/3), int(num % 3)
                       ].set_title(f'{ROBOT_NAMES[num]}')
                    ax[int(num/3), int(num % 3)].set_xlabel('Timestamp')
                    if not num % 3:
                        if not num:
                            ax[0, 0].legend()
                        else:
                            ax2[num].legend()
                        ax[int(num/3), 0].set_ylabel('APE Translation')
                    if num % 3 == 2:
                        ax2[num].set_ylabel('number of poses')

                # plt.draw()
                # plt.pause(1.5)
                plt.savefig(os.path.join(APE_DIR, 'ape.jpg'))
                for num in range(ROBOT_NUM):
                    ax2[num].clear()
                    ax2[num].set_yticks([])
                    ax[int(num/3), int(num % 3)].clear()
                    # ax2[num].line = []
                    # ax[int(num/3), int(num%3)].line = []
                plt.close(fig)

                fig_, ax_ = plt.subplots(2, 3, figsize=(15, 10))
                fig_.clear()
                # SETTINGS.plot_xyz_realistic = False
                for num in range(ROBOT_NUM):
                    if ad_traj_by_label[num]:
                        ax_traj = plot.prepare_axis(
                            fig_, plot.PlotMode.xy, subplot_arg=231+num)
                        ax_traj.set_title(ROBOT_NAMES[num])
                        plot.traj(
                            ax_traj, plot.PlotMode.xy, ad_traj_by_label[num]['est'], label='est', color='blue', plot_start_end_markers=True)
                        plot.traj(
                            ax_traj, plot.PlotMode.xy, ad_traj_by_label[num]['ref'], label='ref', color='green', plot_start_end_markers=True)
                        plot.draw_correspondence_edges(
                            ax_traj, ad_traj_by_label[num]['est'], ad_traj_by_label[num]['ref'], plot.PlotMode.xy, alpha=0.5)
                # plt.draw()
                # plt.pause(1.5)
                plt.savefig(os.path.join(APE_DIR, 'traj.jpg'))
                plt.close(fig_)
                print('-'*10)

                # 计算剩余的睡眠时间
                elapsed_time = time.time() - start_time
                if elapsed_time < INTERVAL:
                    sleep_time = INTERVAL - elapsed_time
                    time.sleep(sleep_time)
                    print(f'Sleeping for {sleep_time} seconds')

        except Exception as e:
            # 输出错误原因和对应的行
            print(
                f'Exiting for {e}, which is in line {sys.exc_info()[-1].tb_lineno}')

            attempt += 1
            if attempt >= retry_count:
                print("Max retry attempts reached. Exiting.")
                plt.close('all')
                for num in range(ROBOT_NUM):
                    ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
                        APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
                    print(f'Saved APE data for {ROBOT_NAMES[num]}')
                break
            else:
                print(f'The {attempt}/{retry_count} Retrying...')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Single mode")
        main(1)
    else:
        print("Distributed mode")
        main()
    # main(1)
