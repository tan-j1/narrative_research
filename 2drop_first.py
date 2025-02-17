import os
import re

def remove_content_before_first(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 记录未找到“First in the tank”的文件
    files_without_first = []

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(source_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()  # 读取整个文件内容
            
            # 使用宽松的正则表达式查找“first in the tank”或“first into the tank”的位置
            pattern = r'first[\s,.]*in[\s,.]*the[\s,.]*tank|first[\s,.]*into[\s,.]*the[\s,.]*tank'
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                first_said_index = match.start()  # 获取匹配的开始位置
                
                # 向上查找“说”字的位置
                say_index = content.rfind('说', 0, first_said_index)  # 找到“说”字的位置
                
                if say_index != -1:
                    # 保留“说”字及其后面的内容
                    new_content = content[say_index:]  # 从“说”字开始保留内容
                    
                    # 生成新的文件名
                    new_filename = f"{os.path.splitext(filename)[0]}_1.txt"
                    new_file_path = os.path.join(target_folder, new_filename)
                    
                    # 写入新文件
                    with open(new_file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f'已处理文件并保存为: {new_filename}')
                else:
                    print("未找到“说”字。")
            else:
                # 记录未找到“First in the tank”的文件
                files_without_first.append(filename)

    # 汇报未找到“First in the tank”的文件
    if files_without_first:
        print("以下文件中未找到“First in the tank”:")
        for file in files_without_first:
            print(file)

# 使用示例
source_folder = r'e:\中科院\shark tank视频下载\raw_data season14'
target_folder = r'E:\中科院\shark tank视频下载\raw_data season14_1'
remove_content_before_first(source_folder, target_folder)
