from flask import render_template
from flask import Flask, redirect, url_for, request, jsonify
import requests
import json
import google_article_search
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
    data = google_article_search.search_article(word, K)
    if len(data['data']) < 1:
        res = '[{"title":"적절한 관련 기사가 없습니다.","content":"empty","highlight":"없네요"}]'
        return res

    response = requests.post("http://210.117.181.115:5003/predict", json=data)
    return_list = []
    return_json = {}
    print("data len : ", len(data['data']))
    print("rep len : ", len(response.json().values()))

    response_list = []

    for doc, ele in zip(data['data'], response.json().values()):
        score, answer = ele
        content = re.sub('\'|\"|”', ' ', doc['paragraphs'][0]['context'])
        url = doc['paragraphs'][0]['qas'][0]['id']
        news = doc['paragraphs'][0]['qas'][0]['answers'][0]['text']

        response_list.append((ele[0], ele[1], content, url, news))

    response_list = sorted(response_list, key=lambda x: x[0], reverse=True)

    for score, answer, content, url, news in response_list:
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
