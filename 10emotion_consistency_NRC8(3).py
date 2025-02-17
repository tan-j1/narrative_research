import pandas as pd
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import jensenshannon

# 加载 NRC 情感词典
def load_nrc_dictionary():
    print("加载 NRC 情感字典...")
    nrc_df = pd.read_excel(r'E:\中科院\shark tank视频下载\NRC-Emotion-Lexicon-v0.92-En_8.xlsx', 
                            header=0)  # 使用第一行作为列名
    
    nrc_dict = {}
    
    for _, row in nrc_df.iterrows():
        word = row['English Word']
        emotions = row[1:]  # 获取情感列
        nrc_dict[word] = Counter()
        
        for emotion, count in zip(emotions.index, emotions):
            try:
                count = int(count)  # 尝试将 count 转换为整数
                if count > 0:
                    nrc_dict[word][emotion] += count
            except ValueError:
                print(f"警告: '{count}' 不是有效的数字，已被忽略。")
    
    print("NRC 情感字典加载完成。")
    return nrc_dict

# 计算情感向量
def calculate_emotion_vector(text, nrc_dict):
    words = text.split()
    emotion_vector = Counter()
    for word in words:
        if word in nrc_dict:
            emotion_vector.update(nrc_dict[word])
    return emotion_vector

# 计算JSD（Jensen-Shannon Divergence）
def calculate_jsd(vec1, vec2):
    # 将情感向量转换为概率分布
    all_emotions = set(vec1.keys()).union(set(vec2.keys()))
    prob1 = np.array([vec1.get(emotion, 0) for emotion in all_emotions], dtype=float)  # 强制转换为float
    prob2 = np.array([vec2.get(emotion, 0) for emotion in all_emotions], dtype=float)  # 强制转换为float
    
    # 将计数转换为概率
    prob1 /= prob1.sum() if prob1.sum() > 0 else 1
    prob2 /= prob2.sum() if prob2.sum() > 0 else 1
    
    # 计算JSD
    return jensenshannon(prob1, prob2) ** 2  # 返回平方后的JSD


# 读取团队成员的内容
def read_team_content(base_paths, file_name, team_members):
    print("读取团队成员内容...")
    members_content = {}
    
    folder_found = False
    for base_path in base_paths:
        csv_dir = os.path.join(base_path, file_name)
        if os.path.exists(csv_dir):
            folder_found = True
            for member_id in team_members:
                file_name = f'说话人_{member_id}.csv'
                file_path = os.path.join(csv_dir, file_name)
                
                if not os.path.exists(file_path):
                    print(f"错误：找不到说话人 {member_id} 的文件")
                    return None
                
                df = pd.read_csv(file_path)
                full_content = ' '.join(df['内容'].astype(str))  # 合并所有内容
                members_content[member_id] = full_content
            break
    
    if not folder_found:
        print(f"错误：在所有路径中都找不到文件夹 {file_name}")
        return None
    
    print("团队成员内容读取完成。")
    return members_content

# 计算情感一致性或多样性
def calculate_emotion(base_paths, file_name, team_members, type="consistency"):
    if len(team_members) <= 1:  # 如果团队成员少于等于1，直接返回None，不做相似度计算
        print("团队成员数量少于等于1，无法计算。")
        return None, None
    
    print(f"计算情感{type}...")
    nrc_dict = load_nrc_dictionary()
    members_content = read_team_content(base_paths, file_name, team_members)
    
    if members_content is None:
        return None, None
    
    # 计算每个成员的情感向量
    emotion_vectors = {}
    for member_id, content in members_content.items():
        emotion_vector = calculate_emotion_vector(content, nrc_dict)
        emotion_vectors[member_id] = emotion_vector
    
    # 根据类型计算一致性或多样性
    results = {}
    for i, member_id in enumerate(emotion_vectors.keys()):
        for other_id in list(emotion_vectors.keys())[i + 1:]:  # 只计算一次
            vector = emotion_vectors[member_id]
            other_vector = emotion_vectors[other_id]
            
            if type == "consistency":
                # 计算余弦相似度
                all_emotions = set(vector.keys()).union(set(other_vector.keys()))
                vector_array = np.array([vector.get(emotion, 0) for emotion in all_emotions]).reshape(1, -1)
                other_vector_array = np.array([other_vector.get(emotion, 0) for emotion in all_emotions]).reshape(1, -1)
                
                similarity = cosine_similarity(vector_array, other_vector_array)[0][0]
                results[(member_id, other_id)] = similarity
            elif type == "diversity":
                # 计算JSD
                jsd_value = calculate_jsd(vector, other_vector)
                results[(member_id, other_id)] = jsd_value
    
    # 计算统计信息
    similarity_values = list(results.values())
    summary = {
        '团队成员数': len(team_members),
        '平均值': np.mean(similarity_values) if similarity_values else 0,
        '最低值': np.min(similarity_values) if similarity_values else 0,
        '最高值': np.max(similarity_values) if similarity_values else 0,
    }

    print("\n统计信息：")
    print(f"团队成员数: {summary['团队成员数']}")
    print(f"平均值: {summary['平均值']}")
    print(f"最低值: {summary['最低值']}")
    print(f"最高值: {summary['最高值']}")

    return results, summary

# 读取并更新表格（根据选择计算一致性或多样性）
def update_emotion(base_paths, input_file, output_file, type="consistency"):
    print("开始读取并更新文件...")
    
    # 根据计算类型选择输出文件名
    if type == "diversity":
        output_file = output_file.replace(".xlsx", "_diversity.xlsx")
    else:
        output_file = output_file.replace(".xlsx", "_consistency.xlsx")

    # 如果输出文件已存在，读取它；否则读取输入文件
    try:
        df = pd.read_excel(output_file)
        print("读取现有输出文件成功")
    except FileNotFoundError:
        df = pd.read_excel(input_file)
        print("输出文件不存在，读取输入文件")
    
    # 新建列来存储情感一致性或多样性
    new_column = f'emotion_{type}(NRC8)'
    df[new_column] = None

    # 遍历每行数据
    for index, row in df.iterrows():
        print(f"\n处理第 {index+1} 行，文件名：{row['file_name']}")
        # 获取团队成员
        team_members = row['team']
        team_members = team_members.strip('[]').split(',')  # 去掉方括号并用逗号分隔
        team_members = [int(member.strip()) for member in team_members]  # 转换为整数
        
        # 计算情感一致性或多样性
        results, summary = calculate_emotion(base_paths, row['file_name'], team_members, type)
        
        # 如果存在结果，取其中的平均值
        if results:
            avg_value = summary['平均值']
            df.at[index, new_column] = avg_value
        else:
            df.at[index, new_column] = None  # 若没有结果，设置为None


    # 保存修改后的表格
    df.to_excel(output_file, index=False)
    print(f"结果已保存至 {output_file}")


if __name__ == "__main__":
    input_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    output_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'

# 定义所有基础路径
base_paths = [
    r'E:\中科院\shark tank视频下载\processed_data_season13',
    r'E:\中科院\shark tank视频下载\processed_data_season14',
    r'E:\中科院\shark tank视频下载\processed_data_season15'
]

# 更新一致性结果
update_emotion(base_paths, input_file, output_file, type="consistency")




