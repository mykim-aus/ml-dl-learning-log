# Iris 분류 신경망 (NumPy, 바닥부터)
# Iris classification neural net (NumPy, from scratch)
# XOR 뼈대(../backprop-numpy/backprop.py)를 3-클래스 실전 데이터로 확장
# Extends the XOR skeleton (../backprop-numpy/backprop.py) to a real 3-class dataset
# KIM이 질문에 답하며 한 단계씩 채워나가는 파일
# KIM fills this in one step at a time, answering questions along the way

import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터 로드 (Iris) — 행 150 샘플, 열 4 특징
#    Load data (Iris) — rows = 150 samples, cols = 4 features
data = load_iris()
X = data.data          # (150, 4) 꽃받침·꽃잎 길이·너비, cm / sepal & petal length·width, cm
y = data.target        # (150,) 0=setosa, 1=versicolor, 2=virginica

# 2. 섞고 train/test 분리 / Shuffle, then split into train/test
#    ⚠️ Iris는 클래스 순서대로 정렬됨(앞50 setosa·중50 versicolor·뒤50 virginica).
#       Iris is sorted by class (first 50 setosa, mid 50 versicolor, last 50 virginica).
#       안 섞고 자르면 test가 한 클래스로 쏠림 → 먼저 섞는다.
#       Cutting without shuffling makes test a single class → shuffle first.
rng = np.random.RandomState(0)
idx = rng.permutation(len(X))          # 0~149를 랜덤 순서로 / 0..149 in random order
X, y = X[idx], y[idx]                   # X와 y를 "같은 순서"로 섞기 / shuffle X and y in the SAME order
n_train = 120                          # 120 학습 / 30 테스트 (8:2) / 120 train / 30 test
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# 3. 정규화 (표준화) — 평균·표준편차는 "학습 데이터로만" 계산 (노트 130·139)
#    Normalize (standardize) — compute mean/std from TRAIN data only (notes 130·139)
#    test는 "안 본 데이터"여야 하므로 train 통계를 test에도 적용 (leakage 방지)
#    test must stay "unseen", so apply the train stats to test too (prevents leakage)
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# 4. 가중치 초기화 (딱 한 번) — 입력 4 → 은닉 8 → 출력 3
#    Init weights (once) — input 4 → hidden 8 → output 3
np.random.seed(42)
W1 = np.random.randn(4, 8)      # (4, 8)
b1 = np.zeros((1, 8))           # (1, 8)
W2 = np.random.randn(8, 3)      # (8, 3)
b2 = np.zeros((1, 3))           # (1, 3)

# 5. 정답을 one-hot으로 — 0→[1,0,0], 1→[0,1,0], 2→[0,0,1] (노트 135·141)
#    Labels to one-hot — 0→[1,0,0], 1→[0,1,0], 2→[0,0,1] (notes 135·141)
y_train_oh = np.eye(3)[y_train]         # (120, 3)
y_test_oh  = np.eye(3)[y_test]          # (30, 3)

# 6. 헬퍼 — 순전파 & 정확도 / Helpers — forward pass & accuracy
def forward(Xb):
    z1 = Xb @ W1 + b1
    a1 = np.maximum(0, z1)                              # ReLU
    z2 = a1 @ W2 + b2
    exp = np.exp(z2 - z2.max(axis=1, keepdims=True))
    out = exp / exp.sum(axis=1, keepdims=True)          # softmax
    return z1, a1, out

def accuracy(Xb, yb):
    _, _, out = forward(Xb)
    pred = out.argmax(axis=1)          # 확률 최대 클래스 / class with max prob (노트/note 135)
    return (pred == yb).mean()         # 맞은 비율 / fraction correct

print("train/test 크기 :", X_train.shape, X_test.shape)
print("test 클래스 분포:", np.bincount(y_test), "(섞어서 세 종류 골고루)")
print(f"학습 전 정확도  : train {accuracy(X_train, y_train):.3f}  test {accuracy(X_test, y_test):.3f}")

# 7. 학습 루프 — 순전파 → 손실 → 역전파 → 업데이트 (역전파는 XOR과 동일 형태, shape만 다름)
#    Training loop — forward → loss → backprop → update (backprop same form as XOR, only shapes differ)
lr = 1.0
epochs = 2000
N = len(X_train)
print("\n학습 시작:")
for epoch in range(epochs):
    z1, a1, output = forward(X_train)                                  # 순전파 / forward
    loss = -np.mean(np.sum(y_train_oh * np.log(output), axis=1))       # 손실 cross-entropy / loss

    dz2 = (output - y_train_oh) / N        # (120,3) 출력 오차 / output error — XOR의 (output-y)/N과 동일 형태
    dW2 = a1.T @ dz2                        # (8,3)
    db2 = dz2.sum(axis=0, keepdims=True)   # (1,3)
    da1 = dz2 @ W2.T                        # (120,8)
    dz1 = da1 * (z1 > 0)                    # (120,8) ReLU 거꾸로 / ReLU backward (켜진 자리만 통과)
    dW1 = X_train.T @ dz1                   # (4,8)
    db1 = dz1.sum(axis=0, keepdims=True)   # (1,8)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 200 == 0:
        print(f"  epoch {epoch:4d}  loss {loss:.4f}"
              f"  train_acc {accuracy(X_train, y_train):.3f}"
              f"  test_acc {accuracy(X_test, y_test):.3f}")

# 최종 결과 — train vs test 정확도 (일반화 확인)
# Final result — train vs test accuracy (generalization check)
print("\n=== 학습 후 ===")
print(f"train 정확도: {accuracy(X_train, y_train):.3f}")
print(f"test  정확도: {accuracy(X_test,  y_test):.3f}")
