import os
from docx import Document

def convert_docx_to_txt(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"创建目标文件夹: {target_folder}")

    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹不存在: {source_folder}")
        return

    # 计数器
    success_count = 0
    error_count = 0

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.docx'):
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename.replace('.docx', '.txt'))
            
            try:
                doc = Document(source_file)
                with open(target_file, 'w', encoding='utf-8') as f:
                    for para in doc.paragraphs:
                        f.write(para.text + '\n')
                
                success_count += 1
                print(f"✅ 成功转换: {filename}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 转换 {filename} 时出错: {str(e)}")

    print(f"\n转换完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")

# 设置路径
source_folder = r"E:\中科院\shark tank视频下载\Shark Tank 第十四季 全22集"
target_folder = r"E:\中科院\shark tank视频下载\raw_data season14"

# 执行转换
convert_docx_to_txt(source_folder, target_folder)