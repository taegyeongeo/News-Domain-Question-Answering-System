
# 뉴스 도메인 질의응답 시스템
본 프로젝트는 `뉴스기사에 대한 질의응답 서비스` 를 제공하기 위해서 진행한 프로젝트입니다. 약 3개월간 ( 21. 03 ~ 21. 05 ) 진행하였으며 Transformer 아키텍쳐 기반의 Encoder를 사용하여 한국어 질의응답 데이터셋으로 fine-tuning을 수행한 모델을 기반으로 최신 뉴스 기사를 기반으로 하여 질의응답 서비스를 제공합니다.

<br>

## 시스템 구성 요소
<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/48018483/120220077-2bff9900-c277-11eb-855f-cadd3221d32f.png" />  
</p>
총 3가지 모듈로 구성되어 있으며 웹 클라이언트에서 질의를 입력받은 후 질문과 유사한 최신 뉴스 기사를 수집하고 이를 기반으로 기계독해를 수행하여 사용자에게 적절한 정답을 제시합니다. 

<br><br>

## **웹 데모 페이지**

작성중

<br>

## Requirements 
### **For model serving**
```
bentoml==0.12.1
torch==1.7.1
attrdict==2.0.1
fastprogress==1.0.0
numpy==1.19.2
transformers==4.1.1
scipy==1.5.4
scikit-learn==0.24.0
seqeval==1.2.2
sentencepiece==0.1.95
six==1.15.0
```
### **For web hosting**
```
conda==4.9.2
Flask==1.1.2
html5lib @ file:///tmp/build/80754af9/html5lib_1593446221756/work
lxml @ file:///tmp/build/80754af9/lxml_1603216285000/work
MarkupSafe==1.1.1
requests @ file:///tmp/build/80754af9/requests_1592841827918/work
urllib3 @ file:///tmp/build/80754af9/urllib3_1603305693037/work
```



## **Model Serving with BentoML**
두가지 MRC모델을 손쉽게 생성가능 <br>
- `make_single_mrc_model.py` : Threshold-based MRC Model 
- `make_dual_mrc_model.py` : Retrospective Reader(IntensiveReadingModule, SketchReadingModule)

### **모델 생성**
```bash
python make_dual_mrc_model.py
```
### **모델 배포**
```bash
bentoml serve DualMRCModel:latest
```



## **데이터셋**
- Korquad2.0, AIHUB 기계독해 데이터셋(뉴스도메인 QAset) 사용
- Korquad2.0은 HTML태그를 제거하고 문단단위로 전처리하여 Squad2.0형식으로 변환
- Negative example을 포함하여 변환된 Korquad2.0 데이터 셋120만개와 AIHUB 기계독해 데이터셋 28만개를 학습시 사용
- 약 7만개의 AIHUB 기계독해 데이터셋을 평가시 사용


## **모델 학습**
- 코쿼드 데이터를 3번, AIHUB 데이터를 7번 반복학습
- 파라미터는 KoELECTRA-small-v3 모델의 configuration을 그대로 사용

## **모델 평가**
- AIHUB 기계독해 데이터셋 35만개의 일부를(20%) 평가 데이터로 사용
- 단일 모델에 대한 평가만 수행
- NoAnswer 분류시 사용하는 임계값을 변경하며 실험

</br>
</br>

|                        | **Total**<br/>(EM) | **Total**<br/>(F1) |  **정답이 있는 경우**<br/>(F1) | **정답이 없는 경우**<br/>(acc) | 
| :--------------------- | :----------------: | :--------------------: | :----------------: | :--------------------: |
 **KoreanNewsQAModel** |    X      |       X        |     **81.84**      |      X       |
  **KoreanNewsQAModel(th=10)** |     **67.87**      |       **82.56**        |     **81.64**      |       **84.85**        |
 **KoreanNewsQAModel(th=0)** |     **70.58**      |       **84.92**        |     **80.53**      |       **95.89**        |

 
 </br>
 전체적인 성능치를 고려하여 임계값을 0으로 설정하여 모델을 서빙하기로 결정
 </br></br>

## Citation
```
@misc{park2020koelectra,
  author = {Park, Jangwon},
  title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/KoELECTRA}}
}
```

## Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [KorQuAD2.0](https://korquad.github.io/)
- [AIHUB 기계독해 데이터셋](https://aihub.or.kr/aidata/86)