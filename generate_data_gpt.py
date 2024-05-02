import json
import random
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# 加載環境變數並創建 OpenAI 客戶端
load_dotenv(".env")
client = OpenAI(
    api_key=os.getenv('openai_api_key'),
)

def get_gpt_reply(content: str) -> str:
    """使用指定的模型發送請求到 OpenAI 並獲取回覆。"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": f"{content}",
            }
        ],
        temperature=0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    resp = response.choices[0].message.content
    return resp

def get_prompt(title: List[str], document: str) -> str:
    """根據提供的標題和文章內容生成prompt。"""
    prompt = '你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，表格可能為害蟲基本資訊或是害蟲防治方法，你的任務是根據此表格生成一組繁體中文的問題及答案。問題必須包含表格內的生物學名，且問題必須針對表格中特定的一格來設計問題，最後將這些問題及答案以範例格式儲存'+'[{"question":"q","answer":"a"}]'\
    +'注意：確認問題是否明確，例如明確指出是詢問哪一個藥劑，或是哪一格資訊。\n\n'\
    + f"類別：{title[0]}\n作物名稱：{title[1]}\n害蟲名稱(基本資訊/防治方法)：{title[2]}\n\n"\
    + f"表格：\n{document}"
    
    return prompt

def load_markdown_data(filename: str) -> List[Dict[str, str]]:
    """從給定的JSON文件中加載Markdown數據，並處理潛在的文件讀取異常。"""
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"{filename}加載成功。")
        return data
    except FileNotFoundError:
        print("錯誤：未找到檔案。")
        return []
    except json.JSONDecodeError:
        print("錯誤：檔案格式不正確。")
        return []
    except Exception as e:
        print(f"發生了意外錯誤：{e}")
        return []

def load_used_documents(*filenames: str) -> set:
    """從多個指定的 JSON 文件中加載已經使用過的文檔。"""
    used_documents = set()
    for filename in filenames:
        try:
            with open(filename, "r", encoding='utf-8') as file:
                data = json.load(file)
                # 更新已使用文檔集合，只添加存在 'document' 鍵的條目
                used_documents.update(doc['document'] for doc in data if 'document' in doc)
        except FileNotFoundError:
            print(f"錯誤：文件 {filename} 未找到。")
        except json.JSONDecodeError:
            print(f"錯誤：文件 {filename} 格式不正確。")
        except Exception as e:
            print(f"發生了意外錯誤：{e}")
    return used_documents

def generate_data(markdown_data: List[Dict[str, str]], max_count: int, used_documents: set = None) -> List[Dict[str, Any]]:
    """從Markdown數據生成數據集，直到達到指定的最大數量，同時確保每個類別平均分配。"""
    data_per_category = max_count // 6  # 每個類別應生成的數據數
    category_counts = {category: 0 for category in ["特作", "蔬菜", "花卉", "果樹", "雜糧", "其他"]}  # 初始化每個類別的計數器
    generated_dataset = []
    total_generated = 0  # 初始化總生成數量

    if used_documents is None:
        used_documents = set()  # 如果沒有提供已使用文檔，則創建一個空集合

    for category in category_counts.keys():
        filtered_data = [data for data in markdown_data if data['title'][0] == category and data["content"] not in used_documents]
        random.shuffle(filtered_data)  # 打亂順序以隨機選取數據
        for data in filtered_data:
            if category_counts[category] >= data_per_category:
                break  # 如果該類別已達到最大數量，則跳出循環
            title = data["title"]
            doc_markdown = data["content"]
            try:
                prompt = get_prompt(title, doc_markdown)
                output = get_gpt_reply(prompt) 
                output_list = parse_output(output, title, doc_markdown)
                generated_count = len(output_list)
                category_counts[category] += generated_count
                total_generated += generated_count  # 更新總生成數量
                generated_dataset.extend(output_list)

            except Exception as e:
                print(f"發生錯誤：{e}")

    for category, count in category_counts.items():
        print(f"類別「{category}」生成了 {count} 筆數據。")
    
    print(f"共生成 {total_generated} 筆數據。")  # 打印總生成數量
    return generated_dataset

def parse_output(output: str, title: List[str], doc_markdown: str) -> List[Dict[str, Any]]:
    """解析GPT回覆的輸出，確保它是list格式，並添加Markdown文檔的標題和內容。"""
    output_list = json.loads(output)
    if isinstance(output_list, list):
        return [{"title": title, "document": doc_markdown, **item} for item in output_list]
    else:
        raise ValueError("輸出不是列表")

def save_dataset(dataset: List[Dict[str, Any]], filename: str) -> None:
    """將生成的數據集保存到JSON文件中，並處理潛在的文件寫入異常。"""
    try:
        with open(filename, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print("數據集保存成功。")  # 文件保存成功時打印消息
    except IOError as e:
        print(f"發生了I/O錯誤：{e}")
    except Exception as e:
        print(f"發生了意外錯誤：{e}")

if __name__ == "__main__":
    filename = "dataset/markdown_only/markdown_data_fix.json"
    markdown_data = load_markdown_data(filename)

    # used_documents_filename = ["dataset/generate/generated_data_200.json", "dataset/generate/generated_data_60_new.json"]
    # used_documents = load_used_documents(used_documents_filename)  # 加載已經使用過的文檔

    max_count = 600  # 定義生成數據的最大數量
    generated_dataset = generate_data(markdown_data, max_count, used_documents=None)

    output_filename = "dataset/generate/generated_data_600.json"
    save_dataset(generated_dataset, output_filename)
