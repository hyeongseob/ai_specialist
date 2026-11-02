# 0. 연습장 — PyTorch 예제 모음

PyTorch 기초부터 GPT까지 순서대로 따라가는 예제입니다. 노트북마다 독립적으로 실행되며 CPU에서 동작합니다.
루트의 `jupyter.bat`으로 Jupyter Lab을 열어서 위에서부터 실행하세요.

| # | 노트북 | 내용 | 관련 교안 |
| --- | --- | --- | --- |
| 01 | [텐서 기초](01_텐서_기초.ipynb) | 생성, dtype, 인덱싱, `dim`, 브로드캐스팅, `view`/`transpose`/`cat`, 행렬곱 | - |
| 02 | [자동미분과 경사하강법](02_자동미분과_경사하강법.ipynb) | `backward`, 연쇄법칙, 선형회귀 직접 구현 → `nn.Linear` + optimizer, 학습 루프 5단계 | 20261102.md |
| 03 | [신경망과 분류](03_신경망과_분류.ipynb) | 활성화 함수, XOR, `nn.Module`, 이진(BCE)/다중(CE) 분류, train/eval | 20261102.md |
| 04 | [Dataset과 DataLoader](04_Dataset과_DataLoader.ipynb) | 커스텀 Dataset, 미니배치, train/val, 슬라이딩 윈도우, padding/mask | Ch2 |
| 05 | [텍스트와 벡터공간](05_텍스트와_벡터공간.ipynb) | 토큰화(글자/단어/BPE), `nn.Embedding`, 코사인 유사도, Skip-gram 학습 | Ch1, Ch2 |
| 06 | [RNN](06_RNN.ipynb) | RNN 수식 ↔ `nn.RNN`, LSTM/GRU, 글자 단위 언어 모델, 기울기 소실 | - |
| 07 | [Seq2Seq와 Attention](07_Seq2Seq와_Attention.ipynb) | Encoder–Decoder, teacher forcing, Attention 유무 비교와 시각화 | - |
| 08 | [Self-Attention과 Multi-Head](08_Self_Attention과_Multi_Head.ipynb) | Scaled dot-product, √d 이유, 위치 정보 부재, Causal Mask, Multi-Head | Ch3 |
| 09 | [Transformer와 GPT](09_Transformer와_GPT.ipynb) | LayerNorm, FeedForward, 잔차 연결, GPT 학습/생성, GPT-2 크기 | Ch4 |

흐름: 텐서 → 미분/학습 → 신경망 → 데이터 → 텍스트를 벡터로 → 순서 처리(RNN) → 병목 해결(Attention) → RNN 제거(Self-Attention) → GPT
