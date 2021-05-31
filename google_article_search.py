import re
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import json

#K = 3
#result_count = 0
# cite_txtfile = open('/home/dblab/news_dic_cite_0430.txt', 'r', encoding='UTF8')             # txt 파일 형식
href_head = "/url?q="
url_sep = "&sa"
headers = {"User-Agent": "Mozilla/5.0"}
# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
#documents = []
#squad_dic = {}
#url_list = []
#news_list = []


def read_dic(dic_loaded):  # txt 파일에서 dic 로 읽어옴
    txtfile = open('/home/dblab/news_dic_cite_0527.txt', 'r', encoding='UTF8')  # txt 파일 형식
    while True:
        line = txtfile.readline()
        if not line:
            break
        tempList = line.split(' : ')
        key = tempList[0]
        value = [tempList[1].split(', ')[0], tempList[1].split(', ')[1], tempList[1].split(', ')[2].rstrip('\n')]
        dic_loaded[key] = value
    # print(dic_loaded)
    txtfile.close()


def clean_str(str_text):
    str_text = str_text.replace('\n', ' ')  # 개행문자 제거
    str_text = ' '.join(str_text.split())  # 다중 공백 제거

    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+)'  # e-mail 주소 제거
    str_text = re.sub(pattern=pattern, repl=' ', string=str_text)
    pattern = '(https|http|ftp)[^)]*[A-Za-z0-9$]+'
    str_text = re.sub(pattern=pattern, repl=' ', string=str_text)

    str_text = re.sub(r'\[[^]]*\]', '', str_text)  # [ * ] 형식 제거
    str_text = re.sub(r'\<[^]]*\>', '', str_text)  # < * > 형식 제거
    str_text = str_text.replace('\\', '')
    # str_text = re.sub("●", '', str_text)
    # str_text = re.sub("■", "", str_text)
    pattern = '[●|◇|▲|◆|■|▶|◀|▷|◉|ⓒ]'
    str_text = re.sub(pattern=pattern, repl='', string=str_text)
    # str_text = re.sub("▶", "",str_text)
    #if "▶" in str_text:  # 뒤에 등장하는 참조태그 제거
    #    str_text = str_text.split("▶")[0]
    if "참고한 사이트 자료" in str_text:
        str_text = str_text.split("참고한 사이트 자료")[0]

    return str_text


def extract_text(cite_URL, cite_class, dic_loaded, documents, news_list, url_list, result_count):  # 해당 사이트가 news_dic에 있을 떄 호출됨, 뉴스 기사를 text 로 크롤링

    response = requests.get(cite_URL, headers=headers)
    value_list = dic_loaded[cite_class]
    tag = value_list[0]
    decoding = value_list[1]

    bs = BeautifulSoup(response.content.decode(decoding, 'replace'), 'lxml')  # 'html.parser'
    text_body = bs.select(tag)
    # print("tag:",tag, " / decoding: ", decoding)

    str_text = ""
    if len(text_body) >= 1:
        for ele in text_body:
            str_text += ele.text
        if len(str_text) <= 10:
            return result_count
        result_count += 1  # 찾은 문서개수 + 1
        str_text = clean_str(str_text)  # 전처리
        documents.append(str_text)  # 발견한 context 를 추가
        url_list.append(cite_URL)  # 해당 기사의 url 추가
        news_list.append(value_list[2])
        print("Find : ",cite_URL) 
    else:
        str_text = "There is no text"
    
    return result_count
    


def find_URL(base_results, cite_dic,  documents, news_list, url_list, result_count, K):  # 현재 페이지에 있는 모든 cite 를 확인해서 news_dic 에 있는 페이지 인지 확인하고 extract_text() 호출
    for result in base_results:
        news_name = result['href']
        # print("here : ",news_name)
        if href_head in news_name:  # "/url?q=" 로 시작하는 url 형식임을 확인
            if 'www.' in news_name:  # 'http://' or 'https://' or 'www.' 형식이 같은 보도사여도 제각각임
                news_name = news_name.split('www.')[1]
                news_name = news_name.split('/')[0]
            elif '//' in news_name:
                news_name = news_name.split('//')[1]
                news_name = news_name.split('/')[0]
            # print(news_name, " --- ",cite_dic)
            if news_name in cite_dic:  # new 보도사가 사전에 존재하면 수행
                cite_URL = get_link(result['href'])
                #print("Find ", cite_URL)
                result_count = extract_text(cite_URL, news_name, cite_dic, documents, news_list, url_list, result_count)
            else:
                cite_URL = get_link(result['href'])  # 해당 사이트 링크
                # print(cite_URL)
                # print(news_name, " 추가 필요 or 뉴스 아님")

            if result_count >= K:
                return result_count
    return result_count


def get_link(news_href):
    cite_link = news_href.split(href_head)[1]
    cite_link = cite_link.split(url_sep)[0]

    cite_link = cite_link.replace("%26", "&")
    cite_link = cite_link.replace("%2F", "/")
    cite_link = cite_link.replace("%3A", ":")
    cite_link = cite_link.replace("%3F", "?")
    cite_link = cite_link.replace("%3D", "=")

    return cite_link


def convert_to_squad(query, documents, news_list, url_list):
    squad_dic = {}
    squad_dic['version'] = ""
    squad_dic['data'] = []

    for idx, document in enumerate(documents):
        data_dic = {}
        data_dic["title"] = "temp_title" + str(idx)
        data_dic["paragraphs"] = []

        paragraphs_dic = {}
        paragraphs_dic["qas"] = []
        paragraphs_dic["context"] = document
        qas_dic = {}
        qas_dic["question"] = query
        qas_dic["answers"] = []
        qas_dic["is_impossible"] = False
        # qas_dic["id"] = idx + 1  # 1부터 순서대로 부여
        qas_dic["id"] = url_list[idx]
        answers_dic = {}
        # answers_dic["text"] = ""
        answers_dic['text'] = news_list[idx]
        answers_dic["answer_start"] = 0

        qas_dic["answers"].append(answers_dic)
        paragraphs_dic["qas"].append(qas_dic)
        data_dic["paragraphs"].append(paragraphs_dic)

        squad_dic["data"].append(data_dic)

    return squad_dic

def search_article(query, topK):
    documents = []
    url_list = []
    news_list = []
    result_count = 0
    page_index = 2
    K = int(topK)
    cite_dic = {}

    read_dic(cite_dic)  # 뉴스 보도사 정보 읽어오기
    baseUrl = 'https://www.google.com/search?q='  # 통합검색 url
    url = baseUrl + quote_plus(query)
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content.decode('utf-8', 'replace'), 'lxml')
    body = soup.select('a')
    #print(body)
    print(url)
    print(page_index - 1, " 페이지 결과")
    result_count = find_URL(body, cite_dic, documents, news_list, url_list, result_count, K)  # 첫 페이지

    # global result_count
    while result_count < K:  # 첫 페이지에서 기사가 충분치 않았다면 순차적으로 페이지 넘김
        # page = "&start=" + str((page_index - 1) * 10)
        # next_page = url + quote_plus(page)
        print("---------------------------------")
        next_page = url + "&start=" + str((page_index - 1) * 10)
        response = requests.get(next_page, headers=headers)
        print(next_page)

        try:
            soup = BeautifulSoup(response.content.decode('utf-8', 'replace'), 'lxml')
            base_results = soup.select('a')
            page_index += 1
            print(page_index - 1, " 페이지 결과")
            # print(base_results)
            result_count = find_URL(base_results, cite_dic, documents, news_list, url_list, result_count, K)
            print()
        except:
            print("모든 페이지 탐색종료, ", (K - result_count), "개 문서 부족")

        if (page_index - 1) > 10:
            squad_dic = convert_to_squad(query, documents, news_list, url_list)
            print("통합검색 종료 - ", (K - result_count), "개 관련문서 부족")
            # save_path = 'searchTest.json'
            # with open(save_path, 'w', encoding='utf-8') as outfile:
            #     json.dump(squad_dic, outfile)
            return squad_dic

    squad_dic = convert_to_squad(query, documents, news_list, url_list,)
    print("통합검색 종료")
    #print(squad_dic)
    return squad_dic

#   save_path = 'searchTest.json'
#   with open(save_path, 'w', encoding='utf-8') as outfile:
#   json.dump(squad_dic, outfile)

#query = "부동산 대책"
#search_article(query,3)
# print(squad_dic)

