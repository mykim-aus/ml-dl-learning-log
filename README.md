<a id="english"></a>
<sub>🌐 **English** · [한국어](#한국어)</sub>

# ML/DL Learning Log

> A personal journal of everything I asked AI while learning machine learning and deep learning. It started on **June 29, 2026** — my very first day studying ML/DL. Every concept here was learned by asking questions, one at a time, until it made sense.

Every note is kept in **both English and Korean**. Each note has a small language switcher at the very top so you can jump between the two versions.

## Entries

| Date | Topic | English | 한국어 |
|---|---|---|---|
| 2026-06-29 | Deep Learning & GPU fundamentals | [Read](en/2026-06-29-deep-learning-and-gpu.md) | [읽기](ko/2026-06-29-딥러닝과-GPU.md) |

## The one-line core

> There are measurements and matching answers; weights start random. Multiply measurement × weight to make a **prediction (forward pass)**, measure how far it is from the answer (**loss**), use **differentiation to get the gradient (direction + size)**, and nudge the weights a little in that direction by the **learning rate**. Run it again with the new weights → repeat until you reach weights with the lowest loss.
>
> This same loop is identical across images, video, text, and LLMs — only three things change per field: ① how input becomes numbers, ② how the answer (loss) is defined, ③ the model architecture.

## How I study

- Learn the principle in notes first → confirm it in code → change the numbers and experiment.
- Fill in the math (calculus, linear algebra) at the exact moment I get stuck, not before.

## Structure

```
ml-dl-learning-log/
├── README.md        ← you are here
├── en/              ← English notes (main)
└── ko/              ← Korean notes (한국어)
```

---

<a id="한국어"></a>
<sub>[English](#english) · 🌐 **한국어**</sub>

# 머신러닝·딥러닝 학습 기록

> 머신러닝·딥러닝을 처음 공부하면서 AI에게 질문한 모든 것을 기록하는 공간입니다. **2026년 6월 29일**, ML/DL을 처음 공부한 그날부터 시작합니다. 모든 개념은 이해될 때까지 하나씩 질문하며 배운 것들입니다.

모든 노트는 **영어와 한국어** 두 버전으로 제공됩니다. 각 노트 맨 위에 작은 언어 전환 링크가 있어 두 버전을 오갈 수 있습니다.

## 기록

| 날짜 | 주제 | English | 한국어 |
|---|---|---|---|
| 2026-06-29 | 딥러닝·GPU 기초 | [Read](en/2026-06-29-deep-learning-and-gpu.md) | [읽기](ko/2026-06-29-딥러닝과-GPU.md) |

## 한 줄 핵심

> 측정값과 그에 맞는 정답지가 있고, 가중치를 랜덤으로 시작한다. 측정값 × 가중치로 **예측을 만들고(순전파)**, 그 예측이 정답과 얼마나 떨어졌는지 잰다(**손실**). **미분으로 기울기(방향+크기)를 구하고**, 그 방향으로 **학습률만큼** 가중치를 조금 옮긴다. 새 가중치로 다시 돌린다 → 손실이 가장 적은 가중치 근처에 갈 때까지 반복한다.
>
> 이 루프는 이미지·동영상·텍스트·LLM 어디서나 동일하다 — 분야마다 바뀌는 건 ① 입력을 숫자로 바꾸는 방식 ② 정답(손실)을 정하는 방식 ③ 모델 구조뿐.

## 공부 방식

- 노트로 원리 먼저 → 코드에서 확인 → 숫자를 바꿔 실험.
- 수학(미분·선형대수)은 막히는 그 지점에서 그때그때 채운다.
