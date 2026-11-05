# AI Specialist 코스웍 실습

AI Specialist 과정(2026.11.2 ~ 11.17)의 강의 교안, 실습 노트북, 개인 연습 코드를 모아 둔 저장소입니다.
일정과 인증평가 출제 포인트는 [course_guide.md](course_guide.md)를 참고하세요.

## 폴더 구조

| 폴더 | 일정 | 내용 |
| --- | --- | --- |
| [0_개인_실습](0_개인_실습/) | - | 직접 작성한 연습용 노트북 |
| [1_언어_모델_이해_및_활용](1_언어_모델_이해_및_활용/) | 11/2 ~ 11/3 | Tokenizer, Dataset, Attention, GPT, 사전학습, 파인튜닝(분류·LoRA·Instruction·DPO) |
| [2_비전_모델_이해_및_활용](2_비전_모델_이해_및_활용/) | 11/4 ~ 11/5 | ResNet18, ViT, DETR, U-Net |
| [3_On-device_AI](3_On-device_AI/) | 11/9 ~ 11/10 | Lightweight CNN, Quantization, Pruning, Knowledge Distillation |
| [4_데이터_모델_활용](4_데이터_모델_활용/) | 11/11 | 데이터 모델, 시계열 분석 |
| [5_AI_모델링_심화](5_AI_모델링_심화/) | 11/12 ~ 11/13 | RAG, NL2SQL, LlamaIndex, Knowledge Graph, GraphRAG, MCP |

각 과목 폴더에는 강의 PDF(`pdf/`)와 실습 노트북이 들어 있습니다.

## 실행 환경

conda 환경 `env_aias_test`(Python 3.12, PyTorch 2.5.1 + CUDA 12.4)를 사용합니다.
패키지 목록은 [environment.yml](environment.yml)에 있습니다.

```bash
# 환경 만들기 (처음 한 번)
conda env create -f environment.yml

# 실행
conda activate env_aias_test
jupyter lab
```

이미 환경이 있으면 `conda env update -f environment.yml --prune`으로 맞출 수 있습니다.

## Git 사용 시 참고

- 노트북은 `nbstripout` 필터로 출력과 메타데이터를 지운 상태로 커밋됩니다([.gitattributes](.gitattributes)).
  새 PC에서 클론했다면 `nbstripout --install`을 한 번 실행하세요.
- 데이터셋(`data/`, `datas/`), 다운로드한 모델(`models/`), 학습 결과(`outputs/`, `*.pth`)는 Git에 올리지 않습니다.
  노트북을 실행하면 다시 내려받거나 만들어집니다.
