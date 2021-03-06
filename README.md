# Robust-spacing
### KLUE RoBERTa-large를 사용한 띄어쓰기 모델
기존의 BERT-base 모델에서 학습량과 레이어의 수가 훨씬 더 많아진 RoBERTa를 사용한 띄어쓰기 보정 리포지토리입니다.  

<a href="https://github.com/twigfarm/letr-sol-spacing">
  <img src="https://img.shields.io/badge/Before Repo-181717?style=flat-square&logo=GitHub&logoColor=white"/>
</a>   

## 바뀌게 된 점
<a href="https://kiwi-carol-258.notion.site/SOL-e13a590cf5f14ae4af32a2a518ef37d7">
  <img src="https://img.shields.io/badge/Open In Notion-FFFFFF?style=flat-square&logo=Notion&logoColor=black"/>
</a>   

* 최대 max_len: 256 → 512
* 사전학습 모델: KoBERT-base → KoRoBERTa-large
* 코드 리팩터링

*** 

```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
