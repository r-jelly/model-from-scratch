<div align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

# Model From Scratch

**RNN에서 Transformer, GPT·BERT, LLaMA, 그리고 SFT·DPO까지 — 언어 모델의 핵심 논문을 직접 구현하고 그 주장을 작은 실험으로 검증하는 프로젝트.**

*"왜 이 구조인가?" 를 코드로 답합니다.*

</div>

---

## 🎯 목표

단순 구현이 아니라 **각 모델이 이전 모델의 어떤 문제를 해결하는지**를 중심으로 진행합니다.

RNN의 한계 → Seq2Seq의 병목 → Attention의 등장 → Transformer의 완성 → GPT/BERT → LLaMA → 추론 최적화 → 효율적 fine-tuning → SFT·DPO까지.

프로젝트를 마치면 다음을 할 수 있어야 합니다.

- **구조:** 각 논문의 핵심 구조를 완성 레이어 없이 구현하고, PyTorch reference와 일치시킨다.
- **주장 검증:** 논문의 주장 하나를 작은 실험으로 확인하고, 결과가 예상과 다르면 그 원인을 설명한다.
- **파이프라인:** 토크나이저 → 사전학습 → KV cache 생성 → LoRA → SFT → DPO를 소형 LM 하나로 처음부터 끝까지 재현한다.

---

## 🧭 Overview

| 항목 | 내용 |
|------|------|
| 기간 | 12단계, 2026년 12월 중순 완료 목표 |
| 환경 | Python 3.11+, PyTorch 2.x, RTX 4070 Ti (12GB) 한 장 |
| 구현 경계 | 구조는 직접, 원시연산은 라이브러리 (완성 레이어는 검증용) |
| 단계마다 하는 일 | 논문 구현 → reference 대조 → 주장 실험 → 결과 README |
| 단계별 결과물 | 구현 파일, `experiment.ipynb`, 목표·구현·결과를 정리한 `README.md` |
| 규모 | 대규모 학습 없이, GPU 한 장으로 학습할 수 있는 소형 모델과 공개 데이터셋 |
| 코드 재사용 | 이전 단계의 블록을 누적 재사용하되, 각 단계의 실험은 독립적으로 실행 가능하게 둔다 |

---

## 🗺 Journey

| 단계 | 폴더 | 논문 | 검증할 주장 | 결과 |
|------|------|------|-------------|------|
| 01 | `01-rnn` | [LSTM (1997)](https://doi.org/10.1162/neco.1997.9.8.1735), [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063) | gating은 gradient 소실을 줄이고, clipping은 gradient 폭발을 막는다 | [README](01-rnn/README.md) |
| 02 | `02-seq2seq` | [Sutskever et al. (2014)](https://arxiv.org/abs/1409.3215), [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) | 고정 context vector는 긴 입력에서 무너지고, attention이 이를 완화한다 | [README](02-seq2seq/README.md) |
| 03 | `03-transformer-block` | [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) §3 | `1/sqrt(d_k)` 스케일링이 없으면 softmax가 포화된다 | [README](03-transformer-block/README.md) |
| 04 | `04-transformer-encdec` | [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762), [BPE (2016)](https://arxiv.org/abs/1508.07909) | recurrence 없이 RNN보다 좋은 번역을 더 빨리 학습한다 | 진행 중 |
| 05 | `05-gpt2-lm` | [GPT-1 (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2 (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 사전학습은 downstream 성능을 높인다 | — |
| 06 | `06-bert-mlm` | [BERT (2019)](https://arxiv.org/abs/1810.04805) | 양방향 사전학습이 left-to-right보다 낫다 | — |
| 07 | `07-modern-llm-blocks` | [LLaMA (2023)](https://arxiv.org/abs/2302.13971) | 같은 예산에서 LLaMA형 블록이 GPT-2형보다 loss가 낮다 | — |
| 08 | `08-kv-cache` | [MQA (2019)](https://arxiv.org/abs/1911.02150), [GQA (2023)](https://arxiv.org/abs/2305.13245) | K/V head 공유가 cache 크기와 latency를 줄인다 | — |
| 09 | `09-lora` | [LoRA (2021)](https://arxiv.org/abs/2106.09685) | 낮은 rank로 full fine-tuning에 근접한다 | — |
| 10 | `10-sft` | [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) | SFT로 base 모델이 지시 형식을 따른다 | — |
| 11 | `11-dpo` | [DPO (2023)](https://arxiv.org/abs/2305.18290) | RM과 RL 없이 선호를 학습한다 | — |
| 12 | `12-integration-lab` | — | 사전학습 → SFT → DPO 미니 LM 통합 | — |

---

## 📁 Structure


```
model-from-scratch/
├── 01-rnn/                    # RNN, LSTM
├── 02-seq2seq/                # Seq2Seq + Bahdanau Attention
├── 03-transformer-block/      # Scaled dot-product / multi-head attention, FFN, LayerNorm
├── 04-transformer-encdec/     # Transformer encoder-decoder, BPE
├── 05-gpt2-lm/                # GPT-1/2 decoder-only LM, byte-level BPE
├── 06-bert-mlm/               # BERT, MLM + NSP
├── 07-modern-llm-blocks/      # LLaMA: RMSNorm, RoPE, SwiGLU
├── 08-kv-cache/               # KV cache, MQA, GQA
├── 09-lora/                   # LoRA
├── 10-sft/                    # Supervised fine-tuning
├── 11-dpo/                    # DPO
├── 12-integration-lab/        # 미니 LM 통합
└── docs/                      # 로드맵, 템플릿 (비공개)
```

각 단계 폴더의 `README.md`에 그 단계의 목표, 구현, 결과를 정리합니다.

---

<div align="center">

Made with ☕ by [R-Jelly](https://github.com/r-jelly)

</div>
