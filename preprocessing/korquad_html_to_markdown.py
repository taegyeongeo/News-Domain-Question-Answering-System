import json
import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup as bs
import html2text

cnt = 0
error = defaultdict(list)

#for file in train_files:
    #print(file){
path = 'D:/translate/korquad'
file_list = os.listdir(path)
#print(file_list)
minus_cnt = 0
b_cnt = 0
file = 'D:/translate/korquad/korquad2.1_train_07.json'
first_time = 0
with open(file, 'r') as f:
    all_json = json.load(f)
    data = all_json['data']

    for qaset in data[:]:
        html_2_text = html2text.HTML2Text()
        context = qaset['context']
        title = qaset['title']
        url = qaset['url']
        raw_html = qaset['raw_html']
        # print(type(context))
        qas = qaset['qas']
        # print(type(qas))
        soup = bs(context, 'lxml')
        a_tag = soup.find('a', text='검색하러 가기')
        div_soup = a_tag.find_next('div').find('div')
        find_index = context.find('<a>검색하러 가기</a>')
        print(find_index)
        print("c: ",context)
        print("d: ",context[find_index+20:])

        sub_context = str(div_soup)
        sub_context = re.sub(r'[/]*', '',  # &gt&lt
                             sub_context).strip()

        new_context = html_2_text.handle(str(div_soup))     # 여기서 answer 이랑 변화가 다른 경우가 생김

        new_context = re.sub(r'[*\'\"《》<>〈〉\(\)‘’{}&]*', '', # &gt&lt
                             new_context).strip()

        new_context = re.sub(r'[\t\n]+', ' ', new_context)

        new_context = new_context.replace('[편집]', '[SEP]')

        new_context = re.sub(' +', ' ',  new_context)
        p = re.compile('([-]+)([|][-]+)*')          # ---|---|--- 같은 형식을 | 로 만듬
        new_context = re.sub(p, '[SEP]', new_context)
        p = re.compile('[|]+([ ]+[|]*)*')  # | | | 같이 반복되면 | 로 바꿈
        new_context = re.sub(p, '| ', new_context)

        h3_split_list = new_context.split('##')
        temp_split_list = h3_split_list[:]
        deleted_list_idx = [False for _ in range(len(temp_split_list))]
        idx = 0
        del_prolog = False
        if h3_split_list[1].find('목차') < 3 and h3_split_list[1].find('목차') >= 0:
            # print(h3_split_list[1])
            del h3_split_list[1]  # 목차 제거 , but 목차가 없는 페이지도 존재하기 때문에 이렇게 제거할 수 없음
            deleted_list_idx[1] = True
            del_prolog = True
        delete_h3 = ["같이 보기[SEP]", "각주 및 참고 문헌[SEP]", "외부 링크[SEP]", "참고 문헌[SEP]"] # "각주[SEP]"
        # "목차",
        result_context = ""
        for ele in h3_split_list:
            add_flag = True
            for delete in delete_h3:
                if delete in ele:
                    add_flag = False
                    deleted_list_idx[idx] = True
                    break
            if del_prolog == True and idx == 0:
                idx += 2
            else:
                idx += 1
            if add_flag:
                result_context += ele

        # result_context = re.sub(' +', ' ',  result_context)
        # p = re.compile('([-]+)([|][-]+)*')          # ---|---|--- 같은 형식을 | 로 만듬
        # result_context = re.sub(p, '[SEP]', result_context)
        # p = re.compile('[|]+([ ]+[|]*)*')  # | | | 같이 반복되면 | 로 바꿈
        # result_context = re.sub(p, '| ', result_context)

        for qa in qas:                      # 큰 1

            #real_flag = False
            answer = qa['answer']['text']   # 1
            question = qa['question']       # 큰 2
            id = qa['id']                   # 큰 3
            html_answer_start = qa['answer']['html_answer_start']   #2
            html_answer_text = qa['answer']['html_answer_text']     #3

            answer_start = qa['answer']['answer_start']             #4


            split_context = context.split(answer)
            #print(split_context[0])
            #print("\n\n")
            #print(split_context[1])

            if first_time == 1:
                html_2_text = html2text.HTML2Text()
                split_context0 = html_2_text.handle(context)
                soup = bs(split_context0, 'lxml')
                a_tag = soup.find('a', text='검색하러 가기')
                div_soup = a_tag.find_next('div').find('div')
                print("0:\n",div_soup)

                html_2_text = html2text.HTML2Text()
                split_context1 = html_2_text.handle(split_context[0])
                print("1:\n",split_context1)
                html_2_text = html2text.HTML2Text()
                split_context2 = html_2_text.handle(split_context[1])
                print("2:\n",split_context2)
            first_time += 1
            html_2_text = html2text.HTML2Text()

            # tt_answer = answer
            new_answer = html_2_text.handle(answer)
            #if real_flag == True:
            #    print(new_answer)
            #    real_flag = False
            new_answer = re.sub(r'[*\'\"《》<>〈〉\(\)‘’{}&]*', '',  # &gt&lt
                   new_answer).strip()

            new_answer = re.sub(r'[\t\n]+', ' ', new_answer)
            new_answer = new_answer.replace('[편집]', '[SEP]')
            new_answer = new_answer.split('##')
            result_answer = ""
            for ele in new_answer:
                result_answer += ele
            result_answer = re.sub(' +', ' ', result_answer)

            #result_answer = re.sub(' +', ' ',  result_answer)
            p = re.compile('([-]+)([|][-]+)*')  # ---|---|--- 같은 형식을 | 로 만듬
            result_answer = re.sub(p, '[SEP]',  result_answer)
            p = re.compile('[|]+([ ]+[|]*)*')  # | | | 같이 반복되면 | 로 바꿈
            result_answer = re.sub(p, '| ', result_answer)

            new_answer = result_answer
            #text0 = str(text0)

            if result_context.find(new_answer) == -1:           # 정답 매칭이 안 되면 error 리스트에 id 넣어줌
                error[file].append(id)                          # 없다면 일단 넣고 수작업으로 수정필요
                cnt += 1
            else:                                               # 있다면 json에 수정해서 포함시킴
                answer_num = 0
                real_answer_order = 0
                pass_flag = False
                answer = answer.strip()

                a = -1
                while True:
                    # a = sub_context.find(answer, a + 1)
                    a = context.find(answer, a + 1)
                    if a == -1:
                        break
                    answer_num += 1
                    if a == answer_start:
                        real_answer_order = answer_num
                temp_answer = answer
                temp_answer = re.sub(r'[/]*', '',  # &gt&lt
                             temp_answer).strip()

                sub_answer_num = 0
                a = -1
                while True:
                    a = sub_context.find(temp_answer, a + 1)
                    if a == -1:
                        break
                    sub_answer_num += 1
                real0 = real_answer_order
                real_answer_order -= (answer_num - sub_answer_num)
                real = real_answer_order
                answer_num_list = [0 for _ in range(len(temp_split_list))]
                count = 0
                answer_IsIn_list = 0
                bb_count = 0
                i_index = []
                i = -1
                for list in temp_split_list:
                    i += 1
                    a = -1
                    first = True
                    while True:
                        a = list.find(new_answer, a + 1)
                        if a == -1:
                            break
                        answer_num_list[i] += 1
                        if deleted_list_idx[1] == True and i == 1 and first == True:
                            real_answer_order -= 1
                            first = False
                            continue
                        count += 1
                        bb_count += 1
                        if count == real_answer_order:  # 진짜 정답의 순서가 발견되면 해당 번째 리스트가 나오는 인덱스를 저장
                            answer_IsIn_list = i
                real2 = real_answer_order
                for b in range(0, answer_IsIn_list+1):
                    if deleted_list_idx[b] == True:              # 만약 정답이 나오는 리스트 앞에서 삭제된 리스트가 있고
                        if b == answer_IsIn_list:                # 찐 정답이 있는 리스트가 삭제되면 에러에 넣어주자
                            b_cnt += 1
                            error[file].append(id)
                            pass_flag = True
                            break
                        if deleted_list_idx[1] == True and b == 1:
                            continue
                        real_answer_order -= answer_num_list[b]  # 그 리스트에도 정답과 같은 문자열이 있었다면 정답의 순번을 땡겨줌
                if pass_flag == False:
                    count = 0
                    new_answer_start = -1

                    a = -1
                    while True:
                        a = result_context.find(new_answer, a + 1)
                        if a == -1:
                            break
                        count += 1
                        if count == real_answer_order:
                            new_answer_start = a
                    if new_answer_start == -1:
                        minus_cnt += 1
                        error[file].append(id)
                    elif new_answer_start == 0:
                        print("\n\n")


print("cnt:",file, " ", cnt)
print("bcnt:",file," ",b_cnt)
print("minus:",file, " ", minus_cnt)