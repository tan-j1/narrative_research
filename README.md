这是一个测度叙事方式的项目

包括了很多原始数据、预处理方案和分析方案

1、通过DownKyi插件下载bilibili视频

2、通过飞书妙计将视频的音频内容按发言人和时序转化为文本

3、创建新环境narrative_research

（Win+R输入cmd

conda create -p xxx\narrative_research python=3.11.8

conda activate xxx

conda deactivate

python --version

conda env list

conda install pytorch==2.3.1 torchvision torchaudio  pytorch-cuda=11.8 -c pytorch -c nvidia

如果要用显卡需要下载CUDA和CuDNN（不同环境通用，下载一次即可，步骤可参照https://blog.csdn.net/weixin_43903639/article/details/132135094?ops_request_misc=&request_id=&biz_id=102&utm_term=cuda12.2%E5%AE%89%E8%A3%85%E6%AD%A5%E9%AA%A4&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-
132135094.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187））

4、通过git bash远程传输了代码和示例

（在本地电脑上：右键点击要上传文件的文件夹选择 "Git Bash Here"，会打开一个命令行窗口

首次使用：

 git config --global user.name "您的GitHub用户名"
 
 git config --global user.email "您的GitHub注册邮箱"

初始化仓库：

 git init

链接远程仓库（URL在GitHub仓库页面的绿色 "Code" 按钮中可以找到）：         

 git remote add origin https://github.com/您的用户名/仓库名.git 

添加文件并上传：  

 git status # 查看文件状态
 
 git add . # 添加所有文件
 
 git commit -m "第一次提交" # 提交更改
 
 git push -u origin main # 推送到GitHub（注意：main是分支名，有些可能是master）

如果想切换分支：      

 git branch -M main  # 将主分支改名为main）

5、代码1-5主要负责原始文件处理

①把文件从docx格式转换为txt格式

②把每集开头的冗余部分去掉

③按照项目分割每集的文件

④更改统一文件名

⑤按时间和说话人将txt文件转换为csv文件

6、代码6-7是自动输入一些用于最终分析的变量数据

⑥输入file_name列

⑦输入expected_investment和expected_stake列

7、手工编码，输入team列、investement列、stake列、sucess列

8、代码8主要负责取对数和标准化因变量valuation

9、代码9-14用于内容一致性、情感一致性、内容多样性和情感多样性的测度

10、代码15用于最后的回归分析和结果（表格结果、热力图结果等）输出
