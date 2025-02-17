import pandas as pd
import os
import re
import ast
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import jensenshannon

# 自定义停用词列表
STOPWORDS = set([ 
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now'
])

def preprocess_text(text):
    """英文文本预处理，使用简单的空格分词"""
    text = str(text)  # 确保文本是字符串类型
    text = text.lower()  # 转换为小写
    text = re.sub(r'[^\w\s]', ' ', text)  # 去除标点符号和特殊字符
    text = re.sub(r'\d+', '', text)  # 去除数字
    words = text.split()  # 简单的空格分词
    filtered_words = [word.strip() for word in words if word.strip() and word.strip() not in STOPWORDS]  # 过滤停用词
    return ' '.join(filtered_words)

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

def analyze_team_content(base_paths, folder_name, team_members, topic_model, type="consistency"):
    """分析指定团队成员的主题一致性或多样性"""
    if len(team_members) <= 1:
        print("团队成员数量少于等于1，无法计算")
        return None
    
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
                processed_contents = [preprocess_text(content) for content in df['内容']]
                full_content = ' '.join(processed_contents)
                members_content[member_id] = full_content
            break  # 找到文件夹后就跳出循环
    
    if not folder_found:
        print(f"错误：在所有路径中都找不到文件夹 {folder_name}")
        return None
    
    texts = [members_content[mid] for mid in team_members]
    print("提取主题...")
    
    # 使用 BERTopic 提取每个成员的主题
    topics, _ = topic_model.transform(texts)
    
    print(f"计算主题{type}...")
    # 获取主题向量
    topic_embeddings = topic_model.embedding_model.embed(texts)
    
    # 根据计算类型选择相似度或多样性计算方法
    if type == "consistency":
        # 使用余弦相似度计算一致性
        similarity_matrix = cosine_similarity(topic_embeddings)
    else:
        # 使用JSD计算多样性
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity_matrix[i, j] = calculate_jsd(topic_embeddings[i], topic_embeddings[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]
    
    results = []
    for i in range(len(team_members)):
        for j in range(i + 1, len(team_members)):
            similarity = similarity_matrix[i, j]
            results.append({
                '说话人1': team_members[i],
                '说话人2': team_members[j],
                '值': similarity
            })
    
    if len(results) == 0:
        print("没有计算出有效的数据")
        return None
    
    avg_value = np.mean([result['值'] for result in results])
    min_value = np.min([result['值'] for result in results]) if results else None
    max_value = np.max([result['值'] for result in results]) if results else None
    
    summary = {
        '团队成员数': len(team_members),
        '成员ID': ', '.join(map(str, team_members)),
        '平均值': avg_value,
        '最低值': min_value,
        '最高值': max_value
    }
    
    print("\n统计信息：")
    print(f"团队成员数: {summary['团队成员数']}")
    print(f"成员ID: {summary['成员ID']}")
    print(f"平均值: {summary['平均值']:.4f}")
    print(f"最低值: {summary['最低值']:.4f}")
    print(f"最高值: {summary['最高值']:.4f}")
    
    return avg_value

def update_topic_consistency(input_file, output_file, base_paths, type="consistency"):
    
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
    
    # 初始化 BERTopic 模型
    topic_model = BERTopic()
    
    # 收集所有文本进行整体训练
    all_texts = []
    for _, row in df.iterrows():
        folder_name = row['file_name']
        
        try:
            team_members = ast.literal_eval(row['team'])
            if not isinstance(team_members, list):
                team_members = [team_members]
        except Exception as e:
            print(f"解析团队成员时出错: {e}")
            team_members = []
        
        # 在所有基础路径中查找文件
        for base_path in base_paths:
            csv_dir = os.path.join(base_path, folder_name)
            if os.path.exists(csv_dir):
                for member_id in team_members:
                    file_name = f'说话人_{member_id}.csv'
                    file_path = os.path.join(csv_dir, file_name)
                    if os.path.exists(file_path):
                        df_member = pd.read_csv(file_path)
                        processed_contents = [preprocess_text(content) for content in df_member['内容']]
                        full_content = ' '.join(processed_contents)
                        all_texts.append(full_content)
                    else:
                        print(f"错误：找不到说话人 {member_id} 的文件")
                break  # 找到文件夹后跳出循环
    
    # 训练 BERTopic 模型
    print("训练 BERTopic 模型...")
    topic_model.fit(all_texts)
    
    # 根据计算类型创建相应的列
    column_name = f'topic_{type}(bertopic)'
    df[column_name] = None
    
    # 计算每个团队的一致性或多样性
    for index, row in df.iterrows():
        folder_name = row['file_name']
        print(f"\n处理文件夹：{folder_name}")
        
        try:
            team_members = ast.literal_eval(row['team'])
            if not isinstance(team_members, list):
                team_members = [team_members]
        except Exception as e:
            print(f"解析团队成员时出错: {e}")
            team_members = []
        
        value = analyze_team_content(base_paths, folder_name, team_members, topic_model, type)
        df.at[index, column_name] = value
    
    
    # 保存修改后的表格
    df.to_excel(output_file, index=False)
    print(f"结果已保存至 {output_file}")

if __name__ == "__main__":
    input_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    output_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    
    base_paths = [
        r'E:\中科院\shark tank视频下载\processed_data_season13',
        r'E:\中科院\shark tank视频下载\processed_data_season14',
        r'E:\中科院\shark tank视频下载\processed_data_season15'
    ]
    
    # 计算一致性
    update_topic_consistency(input_file, output_file, base_paths, type="consistency")
    
    # 计算多样性（取消注释以运行）
    # update_topic_consistency(input_file, output_file, base_paths, type="diversity")




