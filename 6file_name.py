import os
import pandas as pd
import numpy as np

# 定义主目录路径
main_dir_path = r'E:\中科院\shark tank视频下载\processed_data_season13'

# 初始化空列表来存储文件夹名
folder_names = []

# 遍历主目录中的所有文件夹
for folder in os.listdir(main_dir_path):
    folder_path = os.path.join(main_dir_path, folder)
    if os.path.isdir(folder_path):  # 如果是文件夹
        # 提取所需的文件夹名部分
        if '-' in folder:
            # 假设文件夹名格式为 "1-S14E01-480P 清晰-AVC_1"
            #new_name = folder.split('-')[0] + '-' + folder.split('-')[1] + '_' + folder.split('-')[-1][-2:]
            new_name = folder.split('-')[0] + '-' + folder.split('-')[1]
            folder_names.append(new_name)  # 添加文件夹名到列表


# 对文件夹名进行升序排序，先将其转换为数字
folder_names.sort(key=lambda x: (int(x.split('-')[0])))  # 提取数字进行排序

# 读取Excel文件
excel_path = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\最终分析表(4).xlsx'
df = pd.read_excel(excel_path)

# 创建新的DataFrame用于追加
new_rows = pd.DataFrame({'file_name': folder_names})

# 将其他列设置为NaN
for col in df.columns:
    if col != 'file_name':
        new_rows[col] = np.nan

# 追加新的行
df = pd.concat([df, new_rows], ignore_index=True)

# 保存更新后的Excel文件
df.to_excel(excel_path, index=False)
print(f"已在文件末尾追加 {len(folder_names)} 个新文件夹名")
print(f"文件已成功保存到 {excel_path}")
