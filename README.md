# Korean Spacing Model
한국어 RoBERTa를 활용하여 만든 띄어쓰기 모델입니다.
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/fiveflow/roberta-base-spacing)
```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("fiveflow/roberta-base-spacing")
roberta = AutoModelForTokenClassification.from_pretrained("fiveflow/roberta-base-spacing")

org_text = "탄소중립과ESG경영에대한사회적요구확대".replace(" ", "") # 공백제거
label = ["UNK", "PAD", "O", "B", "I", "E", "S"]
# char 단위로 토큰화
token_list = [tokenizer.cls_token_id]
for char in org_text:
    token_list.append(tokenizer.encode(char)[1]) 
token_list.append(tokenizer.eos_token_id)
tkd = torch.tensor(token_list).unsqueeze(0)

output = roberta(tkd).logits

_, pred_idx = torch.max(output, dim=2)
tags = [label[idx] for idx in pred_idx.squeeze()][1:-1]
pred_sent = ""
for char_idx, spc_idx in enumerate(pred_idx.squeeze()[1:-1]):
    # "E" tag 단위로 띄어쓰기
    if label[spc_idx] == "E": pred_sent += org_text[char_idx] + " "
    else: pred_sent += org_text[char_idx]

print(pred_sent.strip())
# '탄소중립과 ESG 경영에 대한 사회적 요구 확대'
```

```bibtex
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
