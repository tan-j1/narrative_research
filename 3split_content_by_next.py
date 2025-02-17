import re
import os

def extract_all_content(file_path):
    # 读取处理后的文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_first_speaker(full_content):
    # 查找第一个说话人
    speaker_pattern = r'说话人 (\d+)\s+(\d+:\d+)\n(.*?)(?=\n说话人 \d+|$)'
    first_speaker_match = re.search(speaker_pattern, full_content, re.DOTALL)
    
    if first_speaker_match:
        first_speaker_number = first_speaker_match.group(1)
        first_speaker_time = first_speaker_match.group(2)
        return first_speaker_number, first_speaker_time
    return None, None

def extract_speaker_content(full_content, speaker_number):
    # 查找指定说话人的对话
    speaker_pattern = rf'说话人 {speaker_number} (\d+:\d+)\n(.*?)(?=\n说话人 \d+|$)'
    speaker_matches = re.finditer(speaker_pattern, full_content, re.DOTALL)
    
    speaker_content = []
    for match in speaker_matches:
        time_stamp = match.group(1)  # 获取时间戳
        content = match.group(2).strip()  # 提取说话人的内容
        position = match.start(2)  # 获取内容在文件中的位置
        speaker_content.append((time_stamp, content, position))
    
    return speaker_content

def split_content_by_next(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    warnings = []  # 用于收集警告信息

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(source_folder, filename)

            # 提取整个文件的内容
            full_content = extract_all_content(file_path)

            # 查找第一个说话人
            first_speaker_number, first_speaker_time = extract_first_speaker(full_content)
            
            if first_speaker_number is None:
                warnings.append(f"文件 {filename} 中未找到第一个说话人。")
                continue

            print(f'文件 {filename} 找到的第一个说话人: 说话人 {first_speaker_number}, 时间戳: {first_speaker_time}')

            # 提取第一个说话人的内容
            speaker_content = extract_speaker_content(full_content, first_speaker_number)

            # 查找“next”的位置
            next_positions = []
            for time_stamp, content, position in speaker_content:
                if "next" in content.lower():  # 检查内容中是否包含“next”或“Next”
                    next_positions.append(position)

            # 划分内容为多个部分
            parts = []
            start_index = 0

            # 将说话人的内容划分为多个部分
            for next_position in next_positions:
                part = full_content[start_index:next_position].strip()
                parts.append(part)
                start_index = next_position  # 更新起始索引为当前“next”的位置

            # 添加最后一部分
            parts.append(full_content[start_index:].strip())

            # 生成新的文件名并写入每一部分
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_count = 0  # 计数输出文件数量
            for i, part in enumerate(parts):
                # 使用指定的命名格式
                new_filename = f"{base_name} 清晰-AVC_{i + 1:02d}.txt"
                new_file_path = os.path.join(target_folder, new_filename)

                # 写入新文件
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(part)
                output_count += 1  # 增加输出文件计数
                print(f'文件 {filename} 已处理并保存为: {new_filename}')

            # 检查输出文件数量
            if output_count != 4:
                warnings.append(f"警告: 文件 {filename} 输出的文件数量为 {output_count}，而不是 4 个。")

    # 最后输出所有警告信息
    if warnings:
        print("\n以下是处理过程中产生的警告信息：")
        for warning in warnings:
            print(warning)

# 使用示例
source_folder = r'e:\中科院\shark tank视频下载\raw_data season13_1'  # 输入文件夹
target_folder = r'e:\中科院\shark tank视频下载\raw_data season13_2'  # 输出文件夹
split_content_by_next(source_folder, target_folder)