import pandas as pd
import re
import os

def extract_speaker_content(file_path, output_dir=None):
    # 设置输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return
    
    # 用于存储每个说话人的数据
    speakers_data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()  # 读取整个文件内容
        
        # 使用更精确的正则表达式
        pattern = r'说话人 (\d+) (\d+:\d+)\n(.*?)(?=说话人 \d+|$)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            speaker, time, text = match.groups()
            if speaker not in speakers_data:
                speakers_data[speaker] = {'time': [], 'content': []}
            
            speakers_data[speaker]['time'].append(time)
            speakers_data[speaker]['content'].append(text.strip())
        
        # 获取输入文件的基本名称（不带扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # 创建子文件夹
        speaker_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(speaker_output_dir, exist_ok=True)

        # 为每个说话人创建CSV文件
        for speaker, data in speakers_data.items():
            df = pd.DataFrame({
                '时间': data['time'],
                '内容': data['content']
            })
            
            output_file = os.path.join(speaker_output_dir, f'说话人_{speaker}.csv')
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f'✅ 已创建文件: {output_file}')
            
        print(f"\n处理完成！")
        print(f"共处理了 {len(speakers_data)} 个说话人的对话")
        
    except Exception as e:
        print(f"❌ 处理文件时出错: {str(e)}")

# 批量处理文件
if __name__ == "__main__":
    input_dir = r'e:\中科院\shark tank视频下载\raw_data season13_3'  # 输入文件夹路径
    output_dir = r'e:\中科院\shark tank视频下载\processed_data_season13'  # 输出文件夹路径



    # 遍历输入文件夹中的所有 TXT 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            extract_speaker_content(file_path, output_dir)  # 调用函数处理每个文件