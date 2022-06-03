# Robust-spacing
### KLUE RoBERTa-large를 사용한 띄어쓰기 모델
기존의 BERT-base 모델에서 학습량과 레이어의 수가 훨씬 더 많아진 RoBERTa를 사용한 띄어쓰기 보정 리포지토리입니다.  
→ [기존의 리포지토리 바로가기]([https://github.com/omry/omegaconf](https://github.com/twigfarm/letr-sol-spacing)
## 바뀌게 된 점
* 최대 max_len: 256 → 512
* 사전학습 모델: KoBERT-base → KoRoBERTa-large
* 코드 리팩터링
