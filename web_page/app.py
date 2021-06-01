from flask import render_template
from flask import Flask, redirect, url_for, request, jsonify
import requests
import json
import google_article_search
import google_article_search_test
import re

app = Flask(__name__)  # init


@app.route('/')  # index
def index():
    return render_template('index_test.html')


@app.route('/search', methods=['GET'])
def search():
    word = request.args.get('word')
    K = request.args.get('K')
    print("Q : ", word)
    print("K : ", K)
    res = ""
    data = {}
    data = google_article_search_test.search_article(word, K)
    if len(data['data']) < 1:
        res = '[{"title":"적절한 관련 기사가 없습니다.","content":"empty","highlight":"없네요"}]'
        return res

    response_intensive = requests.post("http://210.117.181.115:5003/intensive_predict", json=data)
    response_sketch = requests.post("http://210.117.181.115:5003/sketch_predict", json=data)

    return_list = []
    return_json = {}
    # print("data len : ",len(data['data']))
    # print("rep len : ",len(response_intensive.json().values()))

    response_list = []

    for doc, intensive_tuple, cls_score in zip(data['data'], response_intensive.json().values(),
                                               response_sketch.json().values()):
        score, diff_score, answer = intensive_tuple
        total_score = (float(diff_score) + float(cls_score)) / 2
        if total_score <= 0:
            print(total_score)
            content = re.sub('\'|\"|”', ' ', doc['paragraphs'][0]['context'])
            url = doc['paragraphs'][0]['qas'][0]['id']
            news = doc['paragraphs'][0]['qas'][0]['answers'][0]['text']

            response_list.append((score, diff_score, answer, content, url, news))

    if len(response_list) < 1:
        res = '[{"title":"적절한 관련 기사가 없습니다.","content":"empty","highlight":"없네요"}]'
        return res

    response_list = sorted(response_list, key=lambda x: x[0], reverse=True)

    for score, diff_score, answer, content, url, news in response_list:
        answer = re.sub('\'|\"|”', ' ', answer)
        return_json['title'] = f'[{round(score * 100, 2)}%] ' + answer
        return_json['content'] = content
        return_json['highlight'] = answer
        return_json['url'] = url
        return_json['news'] = news

        start_index = return_json['content'].find(answer)
        after_len = len(return_json['content']) - start_index - len(answer)
        clen = 200
        if start_index > clen:
            return_json['content'] = return_json['content'][start_index - clen:]
        if after_len > clen:
            return_json['content'] = return_json['content'][0:clen * 2 + len(answer)]

        if answer != "":
            return_list.append(return_json)
        else:
            print("not answer : ", answer)
        return_json = {}

    res = str(return_list).replace('\'', '\"')

    return res


if __name__ == '__main__':  # run
    app.run(host='0.0.0.0', port=8080)