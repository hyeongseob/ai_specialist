# LLM (언어 모델 이해) 학습 노트

> 인증평가(11/17)는 **교안 ipynb 코드를 그대로 출제**하고, 빈칸이나 함수 전체를 직접 타이핑하는 방식입니다.
> 이 노트는 "이론 → Tensor shape 흐름 → 외워서 쓸 코드 → 간단 실습" 순서로 정리했습니다.
> 출처: `1. 언어 모델 이해/pdf/app_sp_slides_20261013.pdf`, `1. 언어 모델 이해/Chapter_1~5_*.ipynb`

## 목차

- [0. 기본 수학 (코드에 바로 쓰이는 것만)](#0-기본-수학-코드에-바로-쓰이는-것만)
- [1. 벡터 공간 (Chapter 1)](#1-벡터-공간-chapter-1)
- [2. 데이터셋: 토크나이저 · 슬라이딩 윈도우 · 임베딩 (Chapter 2)](#2-데이터셋-토크나이저--슬라이딩-윈도우--임베딩-chapter-2)
- [3. Attention (Chapter 3)](#3-attention-chapter-3)
- [4. GPT 모델 조립 (Chapter 4)](#4-gpt-모델-조립-chapter-4)
- [5. 사전학습 Pretraining (Chapter 5)](#5-사전학습-pretraining-chapter-5)
- [부록. 전체 Tensor shape 한눈에 보기](#부록-전체-tensor-shape-한눈에-보기)
- [부록. 빈칸 대비 체크리스트](#부록-빈칸-대비-체크리스트)

---

## 0. 기본 수학 (코드에 바로 쓰이는 것만)

### 0-1. 내적 · 코사인 유사도

| 개념 | 의미 | 수식 |
| --- | --- | --- |
| 내적 (Dot product) | 의미적 유사도 + 벡터 크기. 같은 방향이면 +, 반대면 −, 직교면 0 | `a · b` |
| 코사인 유사도 | 방향만의 유사도 (−1 ~ 1) | `a·b / (‖a‖‖b‖)` |
| 관계 | 단위 벡터(크기 1)에서는 내적 = 코사인 유사도 | |

Attention 점수 `Q @ K^T`는 결국 **토큰끼리의 내적 = 유사도**입니다.

```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([2.0, 4.0, 6.0])

dot = a @ b                                   # 28.0  (torch.dot(a, b)와 동일)
cos = dot / (a.norm() * b.norm())             # 1.0  (같은 방향)
cos2 = torch.nn.functional.cosine_similarity(a, b, dim=0)  # 1.0
```

### 0-2. 행렬 연산 (`*` vs `@`, transpose)

| 연산 | PyTorch | 설명 |
| --- | --- | --- |
| 요소별 곱 | `A * B` | 같은 위치끼리 곱 |
| 요소별 합 | `A + B` | 같은 위치끼리 합 (shape이 다르면 broadcast) |
| 행렬 곱 | `A @ B` | 앞 행렬의 행 · 뒤 행렬의 열 내적. `(m,n) @ (n,p) → (m,p)` |
| 전치 | `A.T`, `A.transpose(0, 1)` | 행과 열을 바꿈. `(2,3) → (3,2)` |

```python
A = torch.tensor([[1., 2.], [3., 4.]])
I = torch.tensor([[1., 0.], [0., 1.]])
A * I   # [[1,0],[0,4]]
A @ I   # [[1,2],[3,4]]

# 3D 이상에서 transpose는 바꿀 두 축을 지정
x = torch.rand(2, 6, 4)           # (b, T, d)
x.transpose(1, 2).shape           # (2, 4, 6)
```

### 0-3. 선형 함수 → 활성화 함수

- 선형층: `z = Wᵀx + b` → PyTorch `nn.Linear(d_in, d_out)` (가중치 shape: `(d_out, d_in)`)
- 활성화 함수: 비선형성을 부여해서 복잡한 패턴을 학습할 수 있게 함
  - Sigmoid `σ(z) = 1/(1+e^{-z})`: 층이 깊어지면 기울기 소실이 생김
  - ReLU `max(0, x)`: 층을 깊게 쌓을 수 있지만 꺼지는 뉴런이 많아짐
  - **GELU** (GPT가 사용): ReLU를 부드럽게 만든 형태라 학습이 더 안정적임 → Chapter 4 코드에서 사용

### 0-4. Softmax (+ Temperature)

- 숫자 나열을 **확률분포**로 바꿈 (모두 양수, 합 = 1). 지수함수라 작은 차이가 크게 벌어짐
- `softmax(z_i) = e^{z_i} / Σ e^{z_j}`
- 클래스가 2개인 Softmax는 Sigmoid와 같음
- **Temperature T**: `softmax(z / T)`. T가 크면 분포가 평평해짐(창의적), T가 작으면 뾰족해짐(결정적)

```python
z = torch.tensor([10., 8., 1.])
torch.softmax(z, dim=-1)          # [0.8807, 0.1192, 0.0001]
torch.softmax(z / 5.0, dim=-1)    # T=5 → [0.5448, 0.3652, 0.0901] (평평해짐)
torch.softmax(z / 0.5, dim=-1)    # T=0.5 → [0.9820, 0.0180, 0.0000] (뾰족해짐)
```

### 0-5. 신경망 학습 3단계 & 경사하강법

1. **Feedforward (순전파)**: 입력 → 예측값
2. **Loss 계산**: 예측값과 실제값의 차이
3. **Backpropagation (역전파)**: Chain Rule로 각 가중치의 기울기를 계산
   - 예: 치타/사자 2배 × 사자/곰 2배 × 곰/사람 1.5배 = 치타/사람 6배
4. **가중치 갱신 (경사하강법)**: `w ← w − α · ∂L/∂w` (α = learning rate)

PyTorch에서는 이 4단계가 각각 아래 코드에 대응합니다 (Chapter 5 학습 루프).

```python
logits = model(x)                 # 1. 순전파
loss = loss_fn(logits, y)         # 2. 손실
loss.backward()                   # 3. 역전파 (gradient 계산)
optimizer.step()                  # 4. 가중치 갱신
optimizer.zero_grad()             #    다음 배치 전에 gradient 초기화 (안 하면 누적됨)
```

### 0-6. Cross Entropy · KL Divergence · Perplexity

- **Cross Entropy**: 정답 토큰의 확률에 로그를 취해 평균을 낸 뒤 −를 붙인 값. 정답 확률이 1에 가까울수록 0에 가까워짐
- **KL Divergence** = Cross Entropy − (데이터 자체의) Entropy → 모델이 틀린 만큼의 벌점
- **Perplexity (PPL)** = `exp(Cross Entropy)`. "매 토큰마다 평균 몇 개 후보 중에서 고민하는가"
  - 완벽한 모델: PPL = 1. K개 중 균등하게 찍는 모델: PPL = K
  - 예: 정답 확률이 0.5, 0.25, 0.125일 때 → PPL = (2×4×8)^(1/3) = 4

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])     # (N=2, C=3)
targets = torch.tensor([0, 1])               # 정답 클래스 인덱스

# 직접 계산
probs = torch.softmax(logits, dim=-1)
target_probs = probs[torch.arange(2), targets]   # 정답 토큰의 확률
manual = -torch.log(target_probs).mean()

# 내장 함수 (logits를 그대로 넣음! softmax 안 함)
loss = F.cross_entropy(logits, targets)
print(manual, loss)            # 같은 값
print(torch.exp(loss))         # perplexity
```

### 0-7. 조건부 확률 · 베이즈 정리와 LLM

- `P(A|B) = P(A∩B) / P(B)`: 분모가 전체가 아니라 **조건을 만족하는 집합**
- 베이즈: `P(A|B) = P(B|A)P(A) / P(B)`. Prior에 Evidence를 더해 Posterior를 얻음
- LLM 관점:
  - 다음 토큰 예측 = `P(w_t | w_<t)`
  - 프롬프트는 새로운 증거 → 답변 분포가 바뀜
  - RAG는 외부 증거를 문맥에 넣어 환각을 줄임

---

## 1. 벡터 공간 (Chapter 1)

### 이론

- **Embedding**: 데이터(단어)를 벡터로 표현한 것
- 비슷한 의미의 단어는 벡터 공간에서 가까이 위치 → 단어 의미를 **계산**할 수 있음
  - `King − Man + Woman ≈ Queen`
- 유사도는 코사인 유사도로 측정

### 핵심 코드 (`gensim`)

```python
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")        # 사전학습 단어 벡터 (50차원)

model.similarity('cat', 'dog')                    # 두 단어의 코사인 유사도
model.most_similar('computer', topn=5)            # 가장 유사한 단어 5개 → [(word, score), ...]
model.doesnt_match(['apple', 'banana', 'orange', 'car'])  # 'car' (그룹에 안 어울리는 단어)
```

### 간단 실습

```python
# 단어 산술: king - man + woman = ?
model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)  # [('queen', ...)]

# 벡터 직접 보기
vec = model['cat']       # numpy array, shape (50,)
```

| 메서드 | 반환 |
| --- | --- |
| `similarity(w1, w2)` | float (코사인 유사도) |
| `most_similar(w, topn=k)` | `[(단어, 점수), ...]` 리스트 |
| `doesnt_match(list)` | 그룹에서 벗어난 단어 하나 (str) |

---

## 2. 데이터셋: 토크나이저 · 슬라이딩 윈도우 · 임베딩 (Chapter 2)

### 이론

**전처리 흐름**: 텍스트 → Tokenizer(토큰 ID) → Token Embedding + Positional Embedding → 모델 입력

**Tokenizer 종류**

| 종류 | 방식 |
| --- | --- |
| BPE (GPT 계열) | 가장 많이 등장하는 쌍을 찾아 어휘에 추가 |
| WordPiece (BERT) | 각각 나올 확률보다 쌍으로 나올 확률이 높으면 추가 |
| Unigram | 최선의 조합을 찾고 쓸모없는 것은 버림. 확률적 샘플링이라 결과가 매번 다를 수 있고 속도가 느림 |

- 자주 쓰이는 단어는 길게 자름 (압축률 향상). 가능하면 의미 단위로 자름
- 한국어는 EXAONE, 중국어는 Qwen 토크나이저가 효율적 (같은 문장을 더 적은 토큰으로 표현)

**Special Token**: 모델을 제어하고 입출력 구조를 정의하는 신호 (예: `<|endoftext|>`, bos/eos/pad). 학습 대상에 포함되지만 Padding Token은 제외. 모델마다 다름

**위치 인코딩**: 토큰 자체에는 위치 정보가 없음 → `Tom ate the meat ≠ The meat ate Tom`

| 방식 | 특징 |
| --- | --- |
| 절대 위치 (학습형) | GPT-2. `nn.Embedding(context_length, emb_dim)`을 더함 → **교안 코드 방식** |
| Sinusoidal | 삼각함수 사용. 낮은 차원은 빠르게, 높은 차원은 천천히 변함 |
| 상대 위치 | 토큰 사이의 거리를 bias로 더함. 캐싱이 어려워 느림 |
| RoPE | 2개씩 짝지어 각도를 다르게 **회전**시킴 (최신 LLM) |

**데이터셋 = Self-Supervised**: 라벨이 따로 없고 텍스트 자체에서 (입력, 정답)을 만듦. 정답은 입력을 **한 칸 오른쪽으로 민 것** (다음 토큰 예측).

```
입력  x: [ 40,  367, 2885, 1464]
정답  y: [367, 2885, 1464, 1807]   ← x를 한 칸 shift
```

### 핵심 코드 ① tiktoken 인코딩/디코딩

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})   # str → list[int]
strings  = tokenizer.decode(integers)                                  # list[int] → str
```

- `allowed_special`을 지정하지 않으면 `<|endoftext|>`가 포함된 텍스트에서 에러가 남
- `<|endoftext|>`의 ID = 50256 (GPT-2 vocab_size = 50257)

### 핵심 코드 ② x / y 만들기

```python
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
```

### 핵심 코드 ③ `GPTDatasetV1` (★ 통째로 외우기)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

포인트:
- `range(0, len(token_ids) - max_length, stride)`: 마지막 target이 범위를 넘지 않도록 `- max_length`
- `stride == max_length`이면 겹침 없음, `stride < max_length`이면 겹침 (데이터 중복)
- `Dataset`을 상속하면 `__len__`, `__getitem__` 두 개를 반드시 구현

### 핵심 코드 ④ `create_dataloader_v1` (★)

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```

- `drop_last=True`: 마지막 배치가 batch_size보다 작으면 버림 (loss가 튀는 것 방지)

```python
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
# [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
```

### 핵심 코드 ⑤ Token Embedding + Positional Embedding

```python
# 작은 예제
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size, output_dim = 6, 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)   # weight shape (6, 3)
token_embeddings = embedding_layer(input_ids)                  # (4, 3) = weight에서 2,3,5,1번 행 조회

# 실제 크기
vocab_size, output_dim = 50257, 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                  stride=max_length, shuffle=False)
inputs, targets = next(iter(dataloader))          # inputs: (8, 4)
token_embeddings = token_embedding_layer(inputs)  # (8, 4, 256)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))     # (4, 256)

input_embeddings = token_embeddings + pos_embeddings               # (8, 4, 256)  ← broadcast
```

- `nn.Embedding(num_embeddings, embedding_dim)`: 첫 인자는 **개수**(vocab 또는 위치 수), 둘째 인자는 **차원**
- Embedding = "ID로 행을 꺼내는 lookup table" (one-hot @ weight와 같음)
- 위치 임베딩의 입력은 `torch.arange(seq_len)` = `[0, 1, 2, 3]`

### Hugging Face 모델 파일 구조

| 파일 | 비유 | 역할 |
| --- | --- | --- |
| `config.json` | 설계도 | 레이어 수, hidden size, head 수 등 모델 구조 |
| `generation_config.json` | 말하기 습관 | max_length, temperature, top_p, eos_token_id |
| `model.safetensors` | 뇌 | 학습된 가중치 (`.bin`보다 빠르고 안전) |
| `vocab.json` / `merges.txt` | 단어장 / BPE 병합 규칙 | |
| `tokenizer.json` | 번역기 | vocab + merges 통합본 (빠름) |
| `tokenizer_config.json` | | 토크나이저 클래스, 특수 토큰 정의 |

```python
from transformers import AutoTokenizer
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ids = qwen.encode("안녕하세요, 반갑습니다!", add_special_tokens=True)
pieces = [qwen.decode([t]) for t in ids]
qwen.eos_token, qwen.eos_token_id      # 특수 토큰 확인 (bos_token, unk_token, pad_token 등)
```

---

## 3. Attention (Chapter 3)

### 이론

- **Attention** = 문맥 파악. 서로 관련된 토큰일수록 점수가 커짐
  - 예: "애플"이 과일인지 회사인지는 주변 토큰을 보고 결정
- 임베딩에 가중치를 곱해 Q, K, V를 만듦

| | 역할 | 비유 |
| --- | --- | --- |
| **Query** | 질문을 던지는 주체 | "내가 무엇을 찾는가" |
| **Key** | 질문에 답하는 대상 | "나는 어떤 단서인가" (어디를 볼지 정하는 검색용) |
| **Value** | 전달할 정보 | "무엇을 가져올지" (실제 내용) |

- **Q와 K를 분리하는 이유**: 같은 벡터를 쓰면 내적이 대칭이라(`Tom·meat = meat·Tom`) 누가 누구를 참고하는지 구분하지 못하고, 항상 자기 자신을 가장 크게 주목함
- **K와 V를 분리하는 이유**: 검색 단서와 전달할 내용이 섞이지 않게 하기 위해

**Scaled Dot-Product Attention**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

- **Scale (`/ √d_k`)**: 차원 d_k가 커지면 내적값의 표준편차가 √d_k만큼 커짐 → 나눠서 분산을 1로 맞춤 (softmax가 한쪽으로 쏠리지 않게 함)
- **Causal Masking**: 현재 토큰이 **미래 토큰을 보지 못하도록** 대각선 위쪽을 `-inf`로 채움 → softmax 후 0
- **Dropout**: softmax 이후 attention weight 일부를 무작위로 0으로 만듦 (과적합 방지)
- **Context vector**: attention weight로 Value를 가중합한 것 → `z(2) = α21·v(1) + α22·v(2) + ...`
- **Multi-head**: 여러 head가 각각 다른 관점에서 문맥을 파악
  - GPT-2 Small: 12 head × 12 Transformer block
  - 초기 Layer는 문맥 파악, 후반 Layer는 다음 토큰 예측에 주력

### Tensor shape 흐름 (CausalAttention)

```
x            : (b, T, d_in)
Q, K, V      : (b, T, d_out)            ← nn.Linear(d_in, d_out)
K.transpose(1,2) : (b, d_out, T)
scores = Q @ Kᵀ  : (b, T, T)            ← 토큰 × 토큰 유사도
mask[:T, :T]     : (T, T)  상삼각(대각선 위) = 1
weights = softmax(scores/√d_out, dim=-1) : (b, T, T)   각 행의 합 = 1
context = weights @ V : (b, T, d_out)
```

### 핵심 코드 ① `CausalAttention` (★ 통째로 외우기)

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

포인트:
- `register_buffer`: 학습 대상(parameter)은 아니지만 `state_dict`에 저장되고 `.to(device)` 때 함께 이동함
- `torch.triu(..., diagonal=1)`: 대각선은 0, **대각선 위만 1** (= 미래 위치)
- `masked_fill_` (밑줄 = in-place). 마스크는 `.bool()`로 바꿔서 넣음
- `[:num_tokens, :num_tokens]`: 입력이 context_length보다 짧을 때 크기를 맞춤
- 순서: **mask → scale + softmax → dropout → @ values**

### 간단 실습: 마스크 눈으로 확인

```python
T = 4
mask = torch.triu(torch.ones(T, T), diagonal=1)
# tensor([[0., 1., 1., 1.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 1.],
#         [0., 0., 0., 0.]])

scores = torch.rand(T, T)
scores.masked_fill_(mask.bool(), -torch.inf)
w = torch.softmax(scores, dim=-1)
print(w)            # 대각선 위 = 0, 각 행 합 = 1
print(w.sum(-1))    # tensor([1., 1., 1., 1.])
```

### 실행 예제

```python
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your
   [0.55, 0.87, 0.66], # journey
   [0.57, 0.85, 0.64], # starts
   [0.22, 0.58, 0.33], # with
   [0.77, 0.25, 0.10], # one
   [0.05, 0.80, 0.55]] # step
)
d_in = inputs.shape[1]   # 3
d_out = 2
batch = torch.stack((inputs, inputs), dim=0)    # (2, 6, 3)

torch.manual_seed(123)
context_length = batch.shape[1]                 # 6
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)                        # (2, 6, 2)
```

### 핵심 코드 ② `MultiHeadAttentionWrapper` (단순 버전)

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

- head를 여러 개 만들어 마지막 차원으로 이어 붙임 → 출력 `(b, T, d_out * num_heads)`
- 위 예제(`d_out=2, num_heads=2`)의 결과: `(2, 6, 4)`
- `nn.ModuleList`를 써야 내부 파라미터가 모델에 등록됨 (파이썬 list를 쓰면 등록 안 됨)

### 핵심 코드 ③ `MultiHeadAttention` (★★ 가장 중요. Chapter 4에서도 그대로 사용)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
```

**MHA shape 흐름 (외워 둘 것)**

```
x                         (b, T, d_in)
W_key(x)                  (b, T, d_out)
.view(b, T, H, hd)        (b, T, H, hd)        d_out = H * hd
.transpose(1, 2)          (b, H, T, hd)        ← head를 배치처럼 앞으로
Q @ K.transpose(2, 3)     (b, H, T, T)         ← 2D가 아니라 4D라서 (2, 3)
weights @ V               (b, H, T, hd)
.transpose(1, 2)          (b, T, H, hd)        ← 원래 순서로
.contiguous().view(b, T, d_out)                ← head 합치기
out_proj                  (b, T, d_out)
```

Wrapper와 비교:

| | `MultiHeadAttentionWrapper` | `MultiHeadAttention` |
| --- | --- | --- |
| 방식 | head마다 CausalAttention 객체를 따로 만들어 concat | Linear 하나로 만든 뒤 `view`로 head를 쪼갬 (병렬, 효율적) |
| 출력 차원 | `d_out * num_heads` | `d_out` |
| out_proj | 없음 | 있음 |
| 전치 | `transpose(1, 2)` | `transpose(2, 3)` (4D) |

자주 헷갈리는 것:
- `head_dim = d_out // num_heads` (`/`가 아니라 `//`)
- `transpose` 다음에 `view`를 하려면 메모리가 연속적이어야 해서 `.contiguous()`가 필요함
- `keys.shape[-1]`은 MHA에서 `head_dim`임 (√head_dim으로 scale)

### 참고: Attention 시각화 (forward hook)

```python
hooks = [block.att.register_forward_hook(make_hook()) for block in model.trf_blocks]
with torch.no_grad():
    model(input_tensor)
for hook in hooks:
    hook.remove()
# hook_fn(module, input, output): input[0]이 att 모듈의 입력 x
```

---

## 4. GPT 모델 조립 (Chapter 4)

### 이론

GPT는 **Transformer의 Decoder만** 사용 (Causal LM: 앞쪽 문맥만 보고 다음 토큰을 맞춤).
BERT는 Encoder만 사용 (Masked LM: 빈칸 채우기, 양방향 문맥).

```
in_idx → tok_emb + pos_emb → dropout
       → [TransformerBlock] × n_layers
       → final_norm → out_head(Linear emb_dim → vocab_size) → logits
```

**TransformerBlock 구조 (Pre-LayerNorm)**

```
x ─┬─ LayerNorm → MultiHeadAttention → Dropout ─(+)─┬─ LayerNorm → FeedForward → Dropout ─(+)→ out
   └──────────────── shortcut ───────────────────┘   └──────────── shortcut ──────────────┘
```

| 구성 요소 | 역할 |
| --- | --- |
| **LayerNorm** | 각 토큰 벡터를 평균 0, 분산 1로 맞춰 학습 안정화. 마지막에 학습 파라미터 γ(scale), β(shift) 적용. ε은 0으로 나누기 방지 |
| RMSNorm (참고) | 평균 빼기를 생략하고 크기(scale)만 맞춤. 최신 LLM이 사용 |
| **GELU** | 부드러운 ReLU. `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` |
| **FeedForward** | "지식의 저장소". emb_dim → **4배** 확장 → GELU → 다시 emb_dim. 차원을 확장해 특징을 세분화 |
| SwiGLU (참고) | Swish 게이트 × value. 최신 LLM FFN에서 사용 |
| **Residual (shortcut)** | 입력을 출력에 더함 → 변화량만 학습하면 되고 기울기 소실을 막음. 그래서 **입력 shape = 출력 shape** 이어야 함 |
| **Dropout** | 과적합 방지 |

**GPT 입출력**

- 모든 Transformer 블록의 입출력 차원이 같음 (블록을 쌓고 residual을 쓰기 위해)
- 마지막 `out_head`에서만 `emb_dim → vocab_size`로 바뀜
- 예: B=1, T=4, d=768, V=50257 → logits `(1, 4, 50257)`
- **마지막 위치의 logits에서 가장 큰 값의 인덱스 = 다음 토큰 ID**

**GPT-2 Small (124M) 설정**

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 단어 집합 크기
    "context_length": 1024,  # 최대 문맥 길이
    "emb_dim": 768,          # 임베딩 차원
    "n_heads": 12,           # 어텐션 헤드 수
    "n_layers": 12,          # 레이어 수
    "drop_rate": 0.1,        # 드롭아웃 비율
    "qkv_bias": False        # Q,K,V 편향 사용 여부
}
```

### 핵심 코드 ① LayerNorm · GELU · FeedForward (★)

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))    # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

포인트:
- LayerNorm: `dim=-1` (마지막 차원 = 임베딩 차원 기준), `keepdim=True` (broadcast용), `unbiased=False` (N으로 나눔)
- `scale`은 ones, `shift`는 zeros로 초기화. `nn.Parameter`로 감싸야 학습됨
- GELU 상수: `2.0 / torch.pi`, `0.044715`, `x³ = torch.pow(x, 3)`

### 간단 실습: LayerNorm 검증

```python
torch.manual_seed(123)
x = torch.randn(2, 5)
ln = LayerNorm(emb_dim=5)
out = ln(x)
print(out.mean(dim=-1))                      # ≈ [0, 0]
print(out.var(dim=-1, unbiased=False))       # ≈ [1, 1]

# 내장 nn.LayerNorm과 비교
print(torch.allclose(out, nn.LayerNorm(5)(x), atol=1e-5))   # True
```

### 핵심 코드 ② TransformerBlock (★)

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
```

- 패턴은 두 번 반복: `shortcut = x → norm → (att | ff) → dropout → x + shortcut`
- MHA에서 `d_in = d_out = emb_dim` (입출력 shape이 같아야 residual 가능)
- config 키 이름 주의: `n_heads`(MHA 인자명은 `num_heads`), `drop_rate`(인자명 `dropout`)

### 핵심 코드 ③ GPTModel (★★)

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

포인트:
- `nn.Sequential(*[...])`: 리스트 앞의 `*`로 풀어서 넣음
- `torch.arange(seq_len, device=in_idx.device)`: 위치 텐서를 입력과 같은 device에 생성
- `out_head`는 `bias=False`
- shape: `in_idx (B, T)` → `tok_embeds (B, T, 768)` + `pos_embeds (T, 768)` → ... → `logits (B, T, 50257)`

### 핵심 코드 ④ `generate_text_simple` (★ Greedy decoding)

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

```
idx          (B, T)
idx_cond     (B, ≤context_size)     ← 뒤에서부터 context_size만큼만 자름
logits       (B, T, V)
logits[:, -1, :]  (B, V)            ← 마지막 위치 = 다음 토큰 예측
idx_next     (B, 1)                 ← keepdim=True라 2차원 유지 → cat 가능
idx          (B, T+1)
```

- softmax는 단조 증가 함수라 argmax 결과가 같으므로 생략해도 됨
- **Autoregressive**: 생성한 토큰을 다시 입력에 붙여서 반복

### 실행 예제

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()                                   # dropout 끔

tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode("Hello, I am")       # [15496, 11, 314, 716]
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # (1, 4)  배치 차원 추가

out = generate_text_simple(model=model, idx=encoded_tensor,
                           max_new_tokens=10,
                           context_size=GPT_CONFIG_124M["context_length"])
decoded_text = tokenizer.decode(out.squeeze(0).tolist())   # 학습 전이라 엉뚱한 문장 출력
```

### 간단 실습: 파라미터 수 · shape 확인

```python
batch = torch.tensor([[6109, 3626, 6100, 345],
                      [6109, 1110, 6622, 257]])      # (2, 4)
print(model(batch).shape)                            # torch.Size([2, 4, 50257])

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,}")   # 163,009,536 (out_head와 tok_emb의 가중치를 공유하면 약 124M)
```

---

## 5. 사전학습 Pretraining (Chapter 5)

### 이론

**학습 과정**: Dataset 준비 → 순전파 → Loss 계산 → 역전파 → 가중치 갱신 → (주기적으로) 검증

| 개념 | 설명 |
| --- | --- |
| Train / Validation 분리 | Validation은 과적합 감지와 하이퍼파라미터 결정에 사용. 교안에서는 텍스트를 **9:1**로 분리 |
| 비용함수 | 수치 예측은 MSE, 분류(다음 토큰 예측)는 **Cross Entropy** |
| Cross Entropy 계산 | 정답 토큰들의 로그 확률 → 평균 → − 부호 |
| 경사하강법 | Loss를 최소화하는 가중치를 찾는 과정. Optimizer가 스스로 조율 |
| Local minimum | 고차원에서는 거의 발생하지 않음. 오히려 너무 평탄해서 방향을 못 찾는 기울기 소실이 문제 |
| **Adam** | Momentum(관성: 방향) + RMSprop(파라미터별 보폭 조절). **AdamW** = Adam + Weight Decay(규제) |
| Epoch | 전체 데이터를 1회 학습 |
| Step (Iteration) | 파라미터를 1회 업데이트 |
| Batch size | Step 1회에 쓰는 데이터 개수. 1,000개 / batch 100 → 1 epoch = 10 step |
| 작은 batch | 불안정하지만 일반화에 유리, 메모리 적음, 시간 증가 |
| 큰 batch | 안정적이지만 과적합 위험, 메모리 많음, 시간 단축 |
| Training vs Inference | Training은 (Masking 덕분에) **모든 위치를 한 번에** 계산, Inference는 Autoregressive로 한 토큰씩 |

**디코딩 전략**

| 방식 | 설명 |
| --- | --- |
| Greedy | 매번 `argmax` → 결정적, 반복적인 문장이 나오기 쉬움 |
| Temperature | `logits / T` 후 softmax → `torch.multinomial`로 샘플링. T↑ 다양, T↓ 결정적 |
| Top-K | 상위 K개만 남기고 나머지는 `-inf` → 그 안에서 확률적으로 선택 |

### Loss 계산 시 shape (★ 가장 많이 묻는 부분)

```
logits         (B, T, V)    →  logits.flatten(0, 1)   (B*T, V)
target_batch   (B, T)       →  target_batch.flatten() (B*T,)
F.cross_entropy(입력 (N, C), 정답 (N,))
```

`cross_entropy`는 `(N, C)` 형태의 입력과 `(N,)` 형태의 정답 인덱스를 요구하므로, 배치와 시퀀스 차원을 합쳐서 평탄화합니다.

### 핵심 코드 ① `calc_loss_batch` (★)

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss
```

### 핵심 코드 ② `calc_loss_loader` · `evaluate_model` (★)

```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

- `loss.item()`: 텐서에서 파이썬 float만 꺼냄 (그래프와 연결이 끊겨 메모리 절약)
- 평가 패턴: **`model.eval()` → `torch.no_grad()` → ... → `model.train()`**
  - `eval()`: Dropout 등을 끔
  - `no_grad()`: gradient 계산을 끔 (메모리 절약, 속도 향상)

### 핵심 코드 ③ `train_model_simple` (★★ 학습 루프)

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

**학습 4줄 (순서까지 암기)**

```python
optimizer.zero_grad()                                           # 1. 이전 gradient 초기화
loss = calc_loss_batch(input_batch, target_batch, model, device)  # 2. 순전파 + loss
loss.backward()                                                 # 3. 역전파
optimizer.step()                                                # 4. 가중치 갱신
```

- `input_batch.numel()`: 원소 개수 = B × T = 이번 배치에서 본 토큰 수
- `global_step = -1`에서 시작 → 첫 step이 0이 되어 `0 % eval_freq == 0`으로 바로 평가

### 핵심 코드 ④ 텍스트 ↔ 토큰 변환 · 샘플 출력

```python
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # (T,) → (1, T)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                           # (1, T) → (T,)
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]          # = context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()
```

- `unsqueeze(0)`는 배치 차원 추가, `squeeze(0)`은 배치 차원 제거
- `model.pos_emb.weight.shape[0]`: 위치 임베딩의 행 개수 = 모델이 처리할 수 있는 최대 길이

### 핵심 코드 ⑤ `generate` (교안 노트북 버전: Greedy + EOS)

```python
def generate(model, idx, max_new_tokens, context_size, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

`generate_text_simple`과 다른 점은 **EOS 토큰이 나오면 중단**하는 것 하나뿐입니다.

**참고: Temperature + Top-K 버전** (`previous_chapters.py`의 `generate`, 슬라이드 이론에 해당)

```python
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]                    # k번째로 큰 값
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values   # 수치 안정화
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)          # 확률적 샘플링
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### 핵심 코드 ⑥ `main`: 데이터 분리 · Optimizer · DataLoader

```python
def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터 (text_data = the-verdict.txt 내용)
    ...

    # 2. 모델 + 옵티마이저
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    # 3. 9:1 분리 후 DataLoader
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # 4. 학습
    tokenizer = tiktoken.get_encoding("gpt2")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    return train_losses, val_losses, tokens_seen, model
```

| | train_loader | val_loader |
| --- | --- | --- |
| 데이터 | `text_data[:split_idx]` | `text_data[split_idx:]` |
| `shuffle` | **True** (매 epoch 섞음) | **False** (평가는 일관되게) |
| `drop_last` | **True** | **False** (검증 데이터를 버리지 않음) |

- `max_length = stride = context_length` → 겹치지 않는 청크
- 하이퍼파라미터: `lr=5e-4`, `num_epochs=10`, `batch_size=2`, `weight_decay=0.1`, `context_length=256` (속도를 위해 1024에서 줄임)

### 핵심 코드 ⑦ 모델 저장 / 로드

```python
torch.save(model.state_dict(), "outputs/model.pth")                 # 저장

model = GPTModel(GPT_CONFIG_124M)                                     # 1. 빈 모델 생성 (CPU)
model.load_state_dict(torch.load("outputs/model.pth", weights_only=True))  # 2. 가중치 로드
model.to(device)                                                      # 3. device로 이동
```

- 저장하는 것은 모델 객체가 아니라 `state_dict()` (파라미터 딕셔너리)
- 로드할 때는 **같은 config로 구조를 먼저 만들고** 가중치를 넣음

### 간단 실습: Loss와 Perplexity 감 잡기

```python
# 학습 전 모델의 loss ≈ ln(50257) ≈ 10.8 (50257개 중 무작위로 찍는 수준)
import math
print(math.log(50257))          # 10.82

torch.manual_seed(123)
model = GPTModel({**GPT_CONFIG_124M, "context_length": 256}).to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
print(train_loss, math.exp(train_loss))   # loss ≈ 11, PPL ≈ 수만 → 학습이 진행되면 감소
```

### 간단 실습: Temperature / Top-K 직접 해 보기

```python
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

# Greedy
torch.argmax(next_token_logits).item()            # 3

# Temperature 샘플링
torch.manual_seed(123)
for T in [0.1, 1.0, 5.0]:
    probs = torch.softmax(next_token_logits / T, dim=0)
    samples = [torch.multinomial(probs, num_samples=1).item() for _ in range(1000)]
    print(T, torch.bincount(torch.tensor(samples), minlength=9))
# T=0.1 → 거의 전부 3, T=5 → 여러 토큰이 고르게 나옴

# Top-K (k=3)
top_logits, top_pos = torch.topk(next_token_logits, 3)   # 값 [6.75, 6.28, 4.51], 위치 [3, 7, 0]
new_logits = torch.where(next_token_logits < top_logits[-1],
                         torch.tensor(float("-inf")), next_token_logits)
torch.softmax(new_logits, dim=0)                          # 상위 3개 외에는 확률 0
```

---

## 부록. 전체 Tensor shape 한눈에 보기

GPT-2 Small 기준 (B=배치, T=토큰 수, d=768, H=12, hd=64, V=50257)

| 단계 | 코드 | shape |
| --- | --- | --- |
| 텍스트 → ID | `tokenizer.encode(text)` | `list[int]` 길이 T |
| 배치 차원 추가 | `torch.tensor(ids).unsqueeze(0)` | `(1, T)` |
| DataLoader 출력 | `inputs, targets` | `(B, T)`, `(B, T)` |
| 토큰 임베딩 | `tok_emb(in_idx)` | `(B, T, d)` |
| 위치 임베딩 | `pos_emb(torch.arange(T))` | `(T, d)` |
| 합 | `tok + pos` | `(B, T, d)` |
| Q/K/V | `W_query(x)` | `(B, T, d)` |
| head 분리 | `.view(B, T, H, hd).transpose(1, 2)` | `(B, H, T, hd)` |
| attention score | `Q @ K.transpose(2, 3)` | `(B, H, T, T)` |
| context | `weights @ V` | `(B, H, T, hd)` |
| head 합치기 | `.transpose(1, 2).contiguous().view(B, T, d)` | `(B, T, d)` |
| FFN 내부 | `Linear(d, 4d)` | `(B, T, 4d)` → `(B, T, d)` |
| TransformerBlock 출력 | | `(B, T, d)` (입력과 같음) |
| logits | `out_head(final_norm(x))` | `(B, T, V)` |
| 다음 토큰 logits | `logits[:, -1, :]` | `(B, V)` |
| 다음 토큰 | `argmax(..., dim=-1, keepdim=True)` | `(B, 1)` |
| loss 입력 | `logits.flatten(0, 1)`, `target.flatten()` | `(B*T, V)`, `(B*T,)` |

---

## 부록. 빈칸 대비 체크리스트

시험에서는 아무 줄이나 비워질 수 있으므로, 아래 항목을 **보지 않고 쓸 수 있는지** 점검합니다.

### 함수/클래스 통째로 쓰기

- [ ] `GPTDatasetV1` (`__init__`, `__len__`, `__getitem__`)
- [ ] `create_dataloader_v1`
- [ ] `CausalAttention`
- [ ] `MultiHeadAttentionWrapper`
- [ ] `MultiHeadAttention` (view/transpose/contiguous 순서 포함)
- [ ] `LayerNorm`, `GELU`, `FeedForward`
- [ ] `TransformerBlock`
- [ ] `GPTModel`
- [ ] `generate_text_simple` / `generate`
- [ ] `calc_loss_batch`, `calc_loss_loader`, `evaluate_model`
- [ ] `train_model_simple`
- [ ] `text_to_token_ids`, `token_ids_to_text`, `generate_and_print_sample`
- [ ] `main`의 optimizer · train/val loader 부분

### 헷갈리기 쉬운 한 줄

| 위치 | 정답 |
| --- | --- |
| tiktoken 인코딩 | `tokenizer.encode(text, allowed_special={"<\|endoftext\|>"})` |
| target 슬라이싱 | `token_ids[i + 1 : i + max_length + 1]` |
| 루프 범위 | `range(0, len(token_ids) - max_length, stride)` |
| 위치 임베딩 입력 | `torch.arange(seq_len, device=in_idx.device)` |
| 마스크 생성 | `torch.triu(torch.ones(context_length, context_length), diagonal=1)` |
| 마스크 적용 | `attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)` |
| scale + softmax | `torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)` |
| 단일 head 전치 | `keys.transpose(1, 2)` |
| MHA 전치 | `keys.transpose(2, 3)` |
| head 합치기 | `context_vec.contiguous().view(b, num_tokens, self.d_out)` |
| LayerNorm 분산 | `x.var(dim=-1, keepdim=True, unbiased=False)` |
| LayerNorm 출력 | `self.scale * norm_x + self.shift` |
| FFN | `Linear(emb, 4*emb) → GELU() → Linear(4*emb, emb)` |
| 블록 쌓기 | `nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])` |
| out_head | `nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)` |
| 마지막 토큰 | `logits[:, -1, :]` |
| greedy | `torch.argmax(logits, dim=-1, keepdim=True)` |
| 이어 붙이기 | `torch.cat((idx, idx_next), dim=1)` |
| loss | `F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())` |
| optimizer | `torch.optim.AdamW(model.parameters(), lr=..., weight_decay=...)` |
| 학습 4줄 | `zero_grad() → loss 계산 → backward() → step()` |
| 평가 3종 | `model.eval()` + `with torch.no_grad():` + 끝나면 `model.train()` |
| 배치 차원 | `unsqueeze(0)` 추가 / `squeeze(0)` 제거 |
| 저장/로드 | `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path, weights_only=True))` |
