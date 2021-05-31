#-*- coding:utf-8 -*-
import json
import random
import re
# 1
# read_file_path1 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_normal_squad_all.json'
# write_file_path = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_normal_squad_all_added.json'
# 2
# read_file_path1 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_noanswer_squad_all.json'
# write_file_path = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_noanswer_squad_all_added.json'
# 3
# read_file_path1 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_normal_squad_all_added.json'
# read_file_path2 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_noanswer_squad_all_added.json'
# write_file_path = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_merge_squad_all_shuffle.json'
# 순으로 각각 생성함

read_file_path1 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_normal_squad_all_copy.json'
read_file_path2 = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_noanswer_squad_all_copy.json'
write_file_path = 'D:/210331_AIHUB_태깅_병합_셔플링/ko_nia_merge_squad_all_shuffle.json'

json_data1 = {}
with open(read_file_path1, "r", encoding='UTF8') as json_file:
    json_data1 = json.load(json_file)

json_data2 = {}
with open(read_file_path2, "r", encoding='UTF8') as json_file:
    json_data2 = json.load(json_file)


#print(json_data['data'][0]['title'])
# print(json_data2['data'][0]['paragraphs'][0]['qas'][0]['question'])

# 1.  nomal_squad에 'is_impossible' = False 추가
# for i in range(0, len(json_data1['data'])):
#     for j in range(0, len(json_data1['data'][i]['paragraphs'])):
#         for c in range(0, len(json_data1['data'][i]['paragraphs'][j]['qas'])):
#             json_data1['data'][i]['paragraphs'][j]['qas'][c]['is_impossible'] = False

# 2.  noanswer_squad에 'is_impossible' = True, 공백 answer 추가
# for i in range(0, len(json_data2['data'])):
#     for j in range(0, len(json_data2['data'][i]['paragraphs'])):
#         for c in range(0, len(json_data2['data'][i]['paragraphs'][j]['qas'])):
#             json_data2['data'][i]['paragraphs'][j]['qas'][c]['answers'] = []
#             json_data2['data'][i]['paragraphs'][j]['qas'][c]['is_impossible'] = True

# 3. 한 쪽의 json 데이터셋으로 'data' 속성(리스트) 을 덧붙이고 랜덤샘플링
json_data1['data'].extend(json_data2['data'])
random.shuffle(json_data1['data'])

with open(write_file_path, 'w', encoding="utf-8") as outfile:
   json.dump(json_data1, outfile, ensure_ascii=False)
