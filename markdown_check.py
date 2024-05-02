import json
import os

def markdown_txt_to_json(folder_path):
    output = []  # 初始化最終輸出列表

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 確認檔案是txt格式
            file_path = os.path.join(folder_path, filename)  # 獲取檔案完整路徑
            with open(file_path, 'r', encoding='UTF-8') as file:
                lines = file.readlines()
            
            # 檢查文件中是否包含"|"
            if any("|" in line for line in lines):
                # 提取第一行並去除末尾的冒號
                parts = lines[0].strip().split(' ', 2)  # 最多分割成三部分
                if parts[-1].endswith("："):
                    parts[-1] = parts[-1][:-1]  # 移除最後一個字符冒號
                
                # 提取文章內容
                content = ''.join(lines[1:])  # 從第二行開始合併為文章內容
                
                # 將結果添加到輸出列表中
                output.append({"title": parts, "content": content})
    
    # 將結果寫入JSON檔案
    with open('markdown_data.json', 'w', encoding='UTF-8') as json_file:
        json.dump(output, json_file, indent=4, ensure_ascii=False)

def markdown_txt_to_json_fix(folder_path):
    output = []  # 初始化最終輸出列表
    valid_options = {"特作", "蔬菜", "花卉", "果樹", "雜糧", "其他"}  # 有效的選項

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 確認檔案是txt格式
            file_path = os.path.join(folder_path, filename)  # 獲取檔案完整路徑
            with open(file_path, 'r', encoding='UTF-8') as file:
                lines = file.readlines()
            
            # 檢查文件中是否包含"|"
            if any("|" in line for line in lines):
                # 提取第一行並去除末尾的冒號
                first_line = lines[0].strip()
                first_two_chars = first_line[:2]  # 提取前兩個字符
                if first_two_chars in valid_options:
                    parts = first_line.split(' ', 2)  # 最多分割成三部分
                    if parts[-1].endswith("："):
                        parts[-1] = parts[-1][:-1]  # 移除最後一個字符冒號
                
                    # 提取文章內容
                    content = ''.join(lines[1:])  # 從第二行開始合併為文章內容
                    
                    # 將結果添加到輸出列表中
                    output.append({"title": parts, "content": content})
    
    # 將結果寫入JSON檔案
    with open('markdown_data_fix.json', 'w', encoding='UTF-8') as json_file:
        json.dump(output, json_file, indent=4, ensure_ascii=False)

if __name__=="__main__":
    # markdown_txt_to_json("dataset/rawdata")
    markdown_txt_to_json_fix("dataset/rawdata")