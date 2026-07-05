# Iris 분류 신경망 (NumPy, 바닥부터)
# XOR 뼈대(../backprop-numpy/backprop.py)를 3-클래스 실전 데이터로 확장
# KIM이 질문에 답하며 한 단계씩 채워나가는 파일

import numpy as np
from sklearn.datasets import load_iris

# ① 데이터 준비 (Iris)
#    행(세로) = 샘플 150개, 열(가로) = 특징 4개
data = load_iris()
X = data.data          # (150, 4): 꽃받침 길이·너비, 꽃잎 길이·너비 (전부 cm)
y = data.target        # (150,)  : 정답 0=setosa, 1=versicolor, 2=virginica

# 확인용 출력 (뭐가 들어왔는지 눈으로)
print("X shape:", X.shape)                 # (150, 4)
print("y shape:", y.shape)                 # (150,)
print("첫 샘플 특징:", X[0])                # 예: [5.1 3.5 1.4 0.2]
print("첫 샘플 정답:", y[0], "=", data.target_names[y[0]])
print("정답 종류 :", np.unique(y), "→", list(data.target_names))

# ② 입력 정규화 (표준화: 각 특징을 평균0·표준1로)
#    특징마다 스케일이 달라(5.1 vs 0.2) → 당김(기울기)이 불공평 → 같은 스케일로 통일
#    ⚠️ 지금은 전체 X로 평균·표준편차 계산 (임시).
#       ⑤에서 "학습 데이터로만" 계산하도록 고칠 예정 (테스트 훔쳐보기 방지, 노트 130)
X = (X - X.mean(axis=0)) / X.std(axis=0)   # 열(특징)마다: (값 − 평균) ÷ 표준편차

print("\n정규화 후:")
print("특징별 평균  :", X.mean(axis=0).round(2))   # [ 0.  0.  0.  0.]
print("특징별 표준편차:", X.std(axis=0).round(2))   # [1. 1. 1. 1.]
print("첫 샘플 특징 :", X[0].round(2))              # 스케일이 서로 비슷해짐

# ③ 가중치 초기화 (딱 한 번) — 입력 4 → 은닉 8 → 출력 3
#    W shape = (들어오는 개수, 나가는 개수),  b shape = (1, 나가는 개수)
np.random.seed(42)              # 재현성
W1 = np.random.randn(4, 8)      # (4, 8): 입력 4 → 은닉 8
b1 = np.zeros((1, 8))           # (1, 8): 은닉 뉴런마다 bias 1개
W2 = np.random.randn(8, 3)      # (8, 3): 은닉 8 → 출력 3 (클래스 3개)
b2 = np.zeros((1, 3))           # (1, 3): 출력 뉴런마다 bias 1개

print("\n가중치 shape:")
print("W1", W1.shape, " b1", b1.shape, " W2", W2.shape, " b2", b2.shape)
