#更改：不再用位置定位，而是用正则表达式定位E和S这样更具有普适性！！！！！！！！！！！！！
import os
import re
import shutil  # 添加shutil模块用于复制文件

def rename_files(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    # 遍历源文件夹
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):  # 确保是txt文件
            try:
                # 使用正则表达式匹配模式
                # (\d+) 匹配开头的数字
                # (S\d+E\d+) 匹配类似 S09E01 的模式
                # (AVC_\d+) 匹配末尾的 AVC_01 格式
                pattern = r'(\d+).*?(S\d+E\d+).*?(AVC_\d+)'
                match = re.search(pattern, filename)
                
                if match:
                    # 提取匹配的组
                    episode_num = match.group(1)  # 第一个数字
                    season_ep = match.group(2)    # S09E01部分
                    
                    # 从文件名末尾提取数字（去掉.txt后缀）
                    suffix_num = filename[:-4][-2:]  # 获取倒数第4个字符之前的最后两个字符
                    suffix = f"_{suffix_num}"
                    
                    # 构建新文件名
                    new_filename = f"{episode_num}-{season_ep}{suffix}.txt"
                    
                    # 构建完整的文件路径
                    source_path = os.path.join(source_folder, filename)
                    target_path = os.path.join(target_folder, new_filename)
                    
                    # 复制并重命名文件
                    shutil.copy2(source_path, target_path)
                    print(f"已复制并重命名: {filename} -> {new_filename}")
                else:
                    print(f"无法匹配文件名格式: {filename}")
                    
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

# 设置源文件夹和目标文件夹路径
source_folder = r'E:\中科院\shark tank视频下载\raw_data season12_2'
target_folder = r'E:\中科院\shark tank视频下载\raw_data season12_3'

# 执行重命名操作
rename_files(source_folder, target_folder)