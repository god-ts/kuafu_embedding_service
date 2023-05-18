import openai
from flask import Flask, request, jsonify
import requests
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import pandas as pd
import logging

logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s')



df = pd.read_csv('data/客服问题统计表 - 信息录入表_embedding_问题.csv', header=0, names=["category", "fact", "question", "answer", "vector"])
df['vector'] = df['vector'].apply(eval).apply(np.array)

openai.api_key = "sk-Nothing"

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    OPENAI_API_KEY = openai.api_key
    url = "http://c.iaiapp.com:3500/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    return response_data['data'][0]['embedding']

def get_sorted_indices(numbers):
    indexed_numbers = list(enumerate(numbers))
    sorted_indices = sorted(indexed_numbers, key=lambda x: x[1])
    return [index for index, number in sorted_indices]

# def chat_gpt(message: list, model='gpt-3.5-turbo'):
#     OPENAI_API_KEY = openai.api_key
#     if model == 'gpt-4':
#         url = "http://c.iaiapp.com:4000/v1/chat/completions"
#     else:
#         url = "http://c.iaiapp.com:3500/v1/chat/completions"
#
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "messages": message,
#         "temperature": 0.2
#     }
#
#     response = requests.post(url, headers=headers, json=payload)
#     result = response.json()['choices'][0]['message']['content'].strip()
#     return result



app = Flask(__name__)
@app.route('/answer', methods=['POST'])
def answer_question():
    request_data = request.get_json()
    logging.info(f'Received request: {request_data}')
    input_system ='''你是夸父炸串招商部的电话招商人员,目的是吸引客户加盟,筛选出有意向的潜在加盟商，并获得对方的口头同意允许招商经理再次电访
    ###
    在打电话时需要遵循的规则：
    1.确保只对客户提供我给出的事实信息。当客户的问题不能被我提供的事实信息解答时，询问客户是否可以让招商经理来介绍详细信息。
    2.不对客户道歉，不说""非常抱歉""、""抱歉""等词语。
    3.时刻维持你作为夸父炸串客户成功专员的身份，如果客户问你是不是机器人、是不是AI、是不是chatGPT等问题，请予以否认并说明你
    4.如果客户明确拒绝，对客户表示感谢并挂断电话。
    ###
    '''
    content = '''根据以下事实内容进行回复
    内容是:'''
    after_user_text = '''(请用口语化的内容回复我.回复内容少于20字)'''

    message = request_data['chat']['message']
    if len(message) >= 3:
        message[-3]["content"] = str(message[-3]["content"]).replace(after_user_text, "")
    req_message = []
    embeddings = []
    input_text = message[-1]["content"]
    input_embedding = get_embedding(input_text)
    distances = distances_from_embeddings(input_embedding, df['vector'].values, distance_metric='cosine')
    for i in get_sorted_indices(distances)[:1]:
        # print(1 - distances[i])
        if 1 - distances[i] >= 0.85:
            embeddings.append(df['fact'][i])
    for d in embeddings:
        content = content + d + '\n'
    req_message.append({"role": "system", "content": input_system})
    if embeddings == []:
        content = ""
    req_message[0]['content'] += content
    message[-1]["content"] = input_text + after_user_text
    req_message += message

    response_data = {
        "status_code": 200,
        'openai_param': {
            "model": "gpt-3.5-turbo",
            "messages": req_message,
            "temperature": 0.2,
            "stream": True,
        }
    }
    logging.info(f'Sending response: {response_data}')
    return jsonify(response_data)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8787)


