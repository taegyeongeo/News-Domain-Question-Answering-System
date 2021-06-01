# News-Domain-Question-Answering-System

## 뉴스 도메인 질의응답 시스템 (21-1학기 졸업 프로젝트)



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

## **모델 평가**
</br>
모델에 대한 평가는 AIHUB 기계독해 데이터셋 35만건중 일부를(20%) 평가 데이터로 사용하였습니다.</br>
정답이 없다고 판별하는 임계값(=threshold)을 기준으로 실험해보았습니다.
</br>
</br>

|                        | **Total**<br/>(EM) | **Total**<br/>(F1) |  **정답이 있는 경우**<br/>(F1) | **정답이 없는 경우**<br/>(acc) | 
| :--------------------- | :----------------: | :--------------------: | :----------------: | :--------------------: |
 **KoreanNewsQAModel** |    X      |       X        |     **81.84**      |      X       |
  **KoreanNewsQAModel(th=10)** |     **67.87**      |       **82.56**        |     **81.64**      |       **84.85**        |
 **KoreanNewsQAModel(th=0)** |     **70.58**      |       **84.92**        |     **80.53**      |       **95.89**        |

 
 </br>
 NoAnswer 분류 성능을 고려한 전체적인 성능은 임계값이 0일때 더 좋았으나 Span predicton의 성능에 비중을 두어 임계값을 10으로 설정하여 서빙했습니다. (추후 수정가능)
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