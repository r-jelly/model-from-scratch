<div align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

# To the Transformer

**RNN에서 Transformer까지 — 시퀀스 모델의 진화를 직접 구현하는 프로젝트.**

*"왜 Transformer인가?" 를 코드로 답합니다.*

</div>

---

## 🎯 목적

단순 구현이 아니라 **각 모델이 이전 모델의 어떤 문제를 해결하는지**를 중심으로 진행합니다.

RNN의 한계 → Seq2Seq의 병목 → Attention의 등장 → Transformer의 완성.  
이 흐름을 코드와 함께 따라가는 것이 이 프로젝트의 목표입니다.

---

## 🗺 Journey

| 단계 | 주제 | 핵심 질문 |
|------|------|-----------|
| 01 | RNN (Vanilla, LSTM) | 순서 정보를 어떻게 기억하는가? |
| 02 | Seq2Seq | 가변 길이 입출력을 어떻게 처리하는가? |
| 03 | Seq2Seq + Attention | 병목 문제를 어떻게 해결하는가? |
| 04 | Transformer | Recurrence 없이 Attention만으로 충분한가? |

---

## 📁 Structure

```
to-the-transformer/
├── 01-rnn/
│   ├── vanilla_rnn.py
│   └── lstm.py
├── 02-seq2seq/
│   └── seq2seq.py
├── 03-seq2seq-attention/
│   ├── bahdanau.py
│   └── luong.py
├── 04-transformer/
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── utils/                 # 공통 모듈
│   ├── tokenizer.py
│   ├── trainer.py
│   ├── dataset.py
│   └── viz.py
└── docs/                  # 블로그 포스팅 초안
```

---

## ⚙️ Environment

```
Python   3.11+
PyTorch  2.x
NumPy    1.x
```

각 단계는 독립적으로 실행 가능하도록 작성합니다.  
별도의 대규모 학습 없이 toy dataset으로 동작을 확인하는 것을 목표로 합니다.

---

<div align="center">

Made with ☕ by [R-Jelly](https://github.com/r-jelly)

</div>
