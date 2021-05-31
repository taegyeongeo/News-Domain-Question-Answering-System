from flask import render_template
from flask import Flask, redirect, url_for, request, jsonify
#import ubuntuCrawling
import requests
import json
import google_article_search
import re

app = Flask(__name__)  # init

@app.route('/')        # index 
def index():
    return render_template('index_test.html')

@app.route('/search', methods = ['GET'])

def search():
    word = request.args.get('word')
    K = request.args.get('K') 
    print("Q : ",word)
    print("K : ",K)
    #K = 5 
    res = ""
    data = {}
    data = google_article_search.search_article(word,K)
    if len(data['data']) < 1:
        res = '[{"title":"적절한 관련 기사가 없습니다.","content":"empty","highlight":"없네요"}]'
        return res
    
    response = requests.post("http://210.117.181.115:5003/predict", json=data)
    return_list = []
    return_json = {}
    print("data len : ",len(data['data']))
    print("rep len : ",len(response.json().values()))
    
    for doc, ele in zip(data['data'], response.json().values()):
        ele = re.sub('\'|\"|”', ' ', ele) 
        return_json['title'] = ele
        return_json['content'] = re.sub('\'|\"|”', ' ', doc['paragraphs'][0]['context'])
        return_json['highlight'] = ele
        return_json['url'] = doc['paragraphs'][0]['qas'][0]['id']
        return_json['news'] = doc['paragraphs'][0]['qas'][0]['answers'][0]['text']
        
        start_index = return_json['content'].find(ele)
        after_len  = len(return_json['content']) - start_index - len(ele)
        clen = 200
        if start_index > clen:
            return_json['content'] = return_json['content'][start_index-clen:]
        if after_len > clen:
            return_json['content'] = return_json['content'][0:clen*2+len(ele)]
        
        if ele != "":
            #print("answer : ",ele)
            return_list.append(return_json)
        else:
            print("not answer : ",ele)
        return_json = {}
    
    res = str(return_list).replace('\'','\"')
    
    return res

if __name__=='__main__':  # run
    app.run(host='0.0.0.0',port = 8080)
    