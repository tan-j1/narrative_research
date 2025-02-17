import os
import shutil

'''
由于并非每个文件夹改名规则都一样（比如不同博主分享的S11），所以把改名独立在了3split_content_by_next.py之外
'''

# 主目录路径（包含多个文件和文件夹路径）
source_folder = r'E:\中科院\shark tank视频下载\raw_data season13_2'  # 输入文件夹
target_folder = r'E:\中科院\shark tank视频下载\raw_data season13_3'  # 输出文件夹


# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有文件或文件夹
for item in os.listdir(source_folder):
    item_path = os.path.join(source_folder, item)

    # 如果是文件直接处理
    if os.path.isfile(item_path) and item.endswith('.txt'):
        # 假设文件名格式为 "1-S14E01-480P 清晰-AVC_1 清晰-AVC_01.txt"
        parts = item.split('-')
        if len(parts) >= 3:  # 确保文件名符合预期格式
            main_part = parts[0] + '-' + parts[1]
            try:
                # 提取编号部分
                file_number = item.split('清晰-AVC_')[-1].split('.')[0].strip()
                new_name = f"{main_part}_{file_number}.txt"
                new_file_path = os.path.join(target_folder, new_name)

                # 复制并重命名文件
                shutil.copy(item_path, new_file_path)
                print(f'文件 {item} 已复制并重命名为: {new_name}')
            except IndexError:
                print(f"警告: 文件名 '{item}' 中未找到编号部分，跳过处理。")
        else:
            print(f"警告: 文件名 '{item}' 不符合预期格式，跳过处理。")

    # 如果是文件夹，则跳过
    elif os.path.isdir(item_path):
        print(f"跳过文件夹项: {item}")
    else:
        print(f"跳过未知类型项: {item}")

