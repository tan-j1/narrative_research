import pandas as pd
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import jensenshannon

# 加载 NRC 两分类情感词典
def load_nrc_dictionary():
    print("加载 NRC 两分类情感字典...")
    nrc_df = pd.read_excel(r'E:\中科院\shark tank视频下载\NRC-Emotion-Lexicon-v0.92-En_2.xlsx', 
                           header=0)  # 使用第一行作为列名
    
    nrc_dict = {}
    
    # 假设词典包含 'Positive' 和 'Negative' 两列
    for _, row in nrc_df.iterrows():
        word = row['English Word']
        positive = row.get('Positive', 0)
        negative = row.get('Negative', 0)
        nrc_dict[word] = {'Positive': 0, 'Negative': 0}
        
        try:
            positive = int(positive)
            negative = int(negative)
            if positive > 0:
                nrc_dict[word]['Positive'] += positive
            if negative > 0:
                nrc_dict[word]['Negative'] += negative
        except ValueError:
            print(f"警告: 正面 '{positive}' 或负面 '{negative}' 不是有效的数字，已被忽略。")
    
    print("NRC 两分类情感字典加载完成。")
    return nrc_dict

# 计算情感向量
def calculate_emotion_vector(text, nrc_dict):
    words = text.split()
    emotion_vector = {'Positive': 0, 'Negative': 0}
    for word in words:
        if word in nrc_dict:
            emotion_vector['Positive'] += nrc_dict[word].get('Positive', 0)
            emotion_vector['Negative'] += nrc_dict[word].get('Negative', 0)
    return emotion_vector

# 读取团队成员的内容
def read_team_content(base_paths, folder_name, team_members):
    print("读取团队成员内容...")
    members_content = {}
    folder_found = False
    
    # 在所有基础路径中查找对应的文件夹
    for base_path in base_paths:
        csv_dir = os.path.join(base_path, folder_name)
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
            break  # 找到文件夹后就跳出循环
    
    if not folder_found:
        print(f"错误：在所有路径中都找不到文件夹 {folder_name}")
        return None
    
    print("团队成员内容读取完成。")
    return members_content

# 计算JSD（Jensen-Shannon Divergence）
def calculate_jsd(vec1, vec2):
    """计算两个分布之间的JSD"""
    # 确保向量是非负的
    vec1 = np.abs(vec1)
    vec2 = np.abs(vec2)
    
    # 处理零向量的情况
    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
        return 1.0  # 如果有一个向量全为0，返回最大差异度1
    
    # 归一化为概率分布
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    
    # 处理可能的数值问题
    vec1 = np.clip(vec1, 1e-10, 1)  # 避免出现0值
    vec2 = np.clip(vec2, 1e-10, 1)  # 避免出现0值
    
    # 重新归一化
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    
    # 计算JSD
    jsd = jensenshannon(vec1, vec2)
    
    # 处理可能的nan值
    if np.isnan(jsd):
        return 1.0
        
    return jsd ** 2  # 返回平方后的JSD

# 计算情感一致性或多样性
def calculate_emotion_consistency(base_paths, folder_name, team_members, type="consistency"):
    if len(team_members) <= 1:  # 如果团队成员少于等于1，直接返回None，不做相似度计算
        print("团队成员数量少于等于1，无法计算。")
        return None, None  # 返回 None 而不是 0
    
    print(f"计算情感{type}...")
    nrc_dict = load_nrc_dictionary()
    members_content = read_team_content(base_paths, folder_name, team_members)
    
    if members_content is None:
        return None, None
    
    # 计算每个成员的情感向量
    emotion_vectors = {}
    for member_id, content in members_content.items():
        emotion_vector = calculate_emotion_vector(content, nrc_dict)
        emotion_vectors[member_id] = emotion_vector
    
    # 计算情感一致性或多样性
    consistency = {}
    for i, member_id in enumerate(emotion_vectors.keys()):
        for other_id in list(emotion_vectors.keys())[i + 1:]:  # 只计算一次
            vector = emotion_vectors[member_id]
            other_vector = emotion_vectors[other_id]
            
            if type == "consistency":
                # 计算余弦相似度
                if vector and other_vector:  # 确保情感向量不为空
                    # 创建统一的情感类别列表
                    emotions = ['Positive', 'Negative']
                    vector_array = np.array([vector.get(emotion, 0) for emotion in emotions]).reshape(1, -1)
                    other_vector_array = np.array([other_vector.get(emotion, 0) for emotion in emotions]).reshape(1, -1)
                    
                    # 处理全零向量的情况
                    if np.linalg.norm(vector_array) == 0 or np.linalg.norm(other_vector_array) == 0:
                        similarity = 0
                    else:
                        similarity = cosine_similarity(vector_array, other_vector_array)[0][0]
                    consistency[(member_id, other_id)] = similarity
                else:
                    consistency[(member_id, other_id)] = 0  # 如果任一向量为空，设置一致性为0
            else:
                # 计算JSD
                emotions = ['Positive', 'Negative']
                vector_array = np.array([vector.get(emotion, 0) for emotion in emotions])
                other_vector_array = np.array([other_vector.get(emotion, 0) for emotion in emotions])
                jsd_value = calculate_jsd(vector_array, other_vector_array)
                consistency[(member_id, other_id)] = jsd_value
    
    # 计算统计信息
    similarity_values = list(consistency.values())
    summary = {
        '团队成员数': len(team_members),
        '平均值': np.mean(similarity_values) if similarity_values else 0,
        '最低值': np.min(similarity_values) if similarity_values else 0,
        '最高值': np.max(similarity_values) if similarity_values else 0,
    }

    print("\n统计信息：")
    print(f"团队成员数: {summary['团队成员数']}")
    print(f"平均值: {summary['平均值']:.4f}")
    print(f"最低值: {summary['最低值']:.4f}")
    print(f"最高值: {summary['最高值']:.4f}")

    return consistency, summary

# 读取自我介绍说话人(2)文件，计算情感一致性并更新
def update_sentiment_consistency(input_file, output_file, base_paths, type="consistency"):
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
    column_name = f'sentiment_{type}(NRC2)'
    df[column_name] = None

    # 遍历每行数据
    for index, row in df.iterrows():
        print(f"\n处理第 {index+1} 行，文件名：{row['file_name']}")
        # 获取团队成员
        team_members = row['team']
        team_members = team_members.strip('[]').split(',')  # 去掉方括号并用逗号分隔
        team_members = [int(member.strip()) for member in team_members]  # 转换为整数
        
        # 计算情感一致性或多样性
        consistency_results, summary = calculate_emotion_consistency(base_paths, row['file_name'], team_members, type)
        
        # 如果存在结果，取其中的平均值
        if consistency_results:
            avg_similarity = summary['平均值']
            df.at[index, column_name] = avg_similarity
        else:
            df.at[index, column_name] = None  # 若没有结果，设置为None

    # 保存修改后的表格
    df.to_excel(output_file, index=False)
    print(f"结果已保存至 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    output_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'

    # 定义所有基础路径
    base_paths = [
        r'E:\中科院\shark tank视频下载\processed_data_season13',
        r'E:\中科院\shark tank视频下载\processed_data_season14',
        r'E:\中科院\shark tank视频下载\processed_data_season15'
    ]

    # 计算多样性
    update_sentiment_consistency(input_file, output_file, base_paths, type="diversity")
