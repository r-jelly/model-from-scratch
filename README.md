<div align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

# Model From Scratch

**RNN에서 최신 LLM/멀티모달/Diffusion/Alignment까지 — 시퀀스 모델의 진화를 12주간 직접 구현하는 프로젝트.**

*"왜 이 구조인가?" 를 코드로 답합니다.*

</div>

---

## 🎯 목적

단순 구현이 아니라 **각 모델이 이전 모델의 어떤 문제를 해결하는지**를 중심으로 진행합니다.

RNN의 한계 → Seq2Seq의 병목 → Attention의 등장 → Transformer의 완성 → 현대 LLM/멀티모달/생성/정렬 기법까지.
이 흐름을 코드와 함께 따라가는 것이 이 프로젝트의 목표입니다.

전체 계획은 [Model From Scratch: 12주 압축 로드맵](docs/roadmap.md)을 따릅니다. 하루 3시간, 주 6일 기준입니다.

---

## 🧩 구현 경계

핵심 수식과 모델 구조는 **직접 구현**하고, 학습에 필요한 범용 기능은 **라이브러리를 활용**합니다.

| 구분 | 내용 |
|------|------|
| 직접 구현 | RNN/LSTM cell, attention, mask, Transformer block, RoPE, RMSNorm, SwiGLU, GQA, KV cache, LoRA, diffusion process, DPO loss |
| 라이브러리 사용 | PyTorch autograd, `nn.Linear`, `nn.Embedding`, optimizer, DataLoader, AMP, 기본 데이터 전처리 |

---

## 🗺 Journey

| 주차 | 폴더 | 주제 | 핵심 질문 |
|------|------|------|-----------|
| 01 | `01-rnn` | RNN & LSTM | 순서 정보를 어떻게 기억하는가? |
| 02 | `02-seq2seq` | Seq2Seq + Bahdanau Attention | 고정 context vector의 병목을 어떻게 해결하는가? |
| 03 | `03-transformer-block` | Transformer 핵심 블록 | Attention 수식을 모듈 단위로 어떻게 옮기는가? |
| 04 | `04-transformer-encdec` | Transformer Encoder-Decoder | Recurrence 없이 Attention만으로 충분한가? |
| 05 | `05-gpt2-lm` | GPT-2형 Decoder-only LM | Causal LM은 어떻게 다음 토큰을 예측하고 생성하는가? |
| 06 | `06-bert-mlm` | BERT & Masked LM | 양방향 문맥은 생성형 모델과 무엇이 다른가? |
| 07 | `07-modern-llm-blocks` | 현대 LLM 블록 | RMSNorm/RoPE/SwiGLU/GQA/KV cache는 무엇을 개선하는가? |
| 08 | `08-lora` | LoRA & 효율적 Fine-tuning | 전체 파라미터를 안 건드리고도 적응할 수 있는가? |
| 09 | `09-vit-clip` | ViT & CLIP | Transformer는 이미지·멀티모달로 어떻게 확장되는가? |
| 10 | `10-diffusion` | Diffusion Model | Noise에서 데이터를 어떻게 복원하는가? |
| 11 | `11-dpo` | DPO 기반 Alignment | 선호 데이터로 응답 확률을 어떻게 조정하는가? |
| 12 | `12-integration-lab` | 통합 프로젝트 | 지금까지의 구현을 재사용 가능한 코드베이스로 어떻게 통합하는가? |

---

## 📁 Structure

```
model-from-scratch/
├── 01-rnn/                    # 1주차: Vanilla RNN, LSTM
├── 02-seq2seq/                # 2주차: Seq2Seq + Bahdanau Attention
├── 03-transformer-block/      # 3주차: Scaled dot-product / multi-head attention, FFN
├── 04-transformer-encdec/     # 4주차: Transformer encoder-decoder
├── 05-gpt2-lm/                # 5주차: GPT-2형 decoder-only LM
├── 06-bert-mlm/               # 6주차: BERT, Masked Language Modeling
├── 07-modern-llm-blocks/      # 7주차: RMSNorm, RoPE, SwiGLU, GQA, KV cache
├── 08-lora/                   # 8주차: LoRA, parameter-efficient fine-tuning
├── 09-vit-clip/               # 9주차: ViT, CLIP
├── 10-diffusion/              # 10주차: DDPM
├── 11-dpo/                    # 11주차: DPO alignment
├── 12-integration-lab/        # 12주차: 통합 실험 코드베이스
├── utils/                     # 공통 모듈 (tokenizer, trainer, dataset, viz)
└── docs/                      # 로드맵, 블로그 포스팅 초안
```

---

## ⚙️ Environment

```
Python   3.11+
PyTorch  2.x
NumPy    1.x
```

각 주차는 독립적으로 실행 가능하도록 작성합니다.
별도의 대규모 학습 없이 toy dataset / 소형 데이터셋으로 동작을 확인하는 것을 목표로 합니다.

---

<div align="center">

Made with ☕ by [R-Jelly](https://github.com/r-jelly)

</div>
