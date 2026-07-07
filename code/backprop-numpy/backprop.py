# 2-layer neural network from scratch (NumPy)
# 순전파 + 역전파를 손으로 직접 구현해보기 / Implement forward + backprop by hand
# KIM이 질문에 답하며 한 줄씩 채워나가는 파일 / KIM fills this in line by line, answering questions

import numpy as np

# 1. 데이터 준비 (XOR) / Data prep (XOR)
#    행(세로) = 샘플 개수, 열(가로) = 특징 개수 / rows = #samples, cols = #features
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])   # (4, 2): 샘플 4개, 특징 2개 / 4 samples, 2 features each

y = np.array([[0],
              [1],
              [1],
              [0]])      # (4, 1): 샘플마다 정답 1개 / one label per sample

# 2. 가중치 초기화 (딱 한 번) / Init weights (once)
#    세트(뉴런) 하나 = 세로 열 하나. 세트끼리 다르게 하려고 랜덤으로 시작.
#    One set (neuron) = one column. Start random so the sets differ from each other.
np.random.seed(42)              # 재현성: 매번 같은 랜덤값 / reproducibility: same randoms each run
W1 = np.random.randn(2, 8)      # (2, 8): 입력 2 → 은닉 뉴런 8 / input 2 → hidden 8
b1 = np.zeros((1, 8))           # (1, 8): 뉴런마다 편향 1개(0으로 시작) / one bias per neuron (start 0)
W2 = np.random.randn(8, 1)      # (8, 1): 은닉 8 → 출력 뉴런 1 / hidden 8 → output 1
b2 = np.zeros((1, 1))           # (1, 1): 출력 편향 / output bias

lr = 1.0                        # 학습률 (한 걸음 크기) / learning rate (step size)
epochs = 10000                  # 반복 횟수 / number of iterations

# 3~6을 반복: 예측 → 손실 → 역전파 → 업데이트
# Repeat 3~6: predict → loss → backprop → update
for epoch in range(epochs):
    # 3. 순전파 / Forward pass
    z1 = X @ W1 + b1                # (4,8) 1층 가중합 / layer-1 weighted sum
    a1 = np.maximum(0, z1)          # (4,8) ReLU (음수→0) / ReLU (negatives→0)
    z2 = a1 @ W2 + b2               # (4,1) 2층 가중합 / layer-2 weighted sum
    output = 1 / (1 + np.exp(-z2))  # (4,1) sigmoid → 0~1 예측 / sigmoid → 0~1 prediction

    # 4. 손실 (BCE) / Loss (BCE)
    loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

    # 5. 역전파 (기울기 구하기) / Backprop (compute gradients)
    dz2 = (output - y) / len(X)     # (4,1) 출력 오차 신호 / output error signal
    dW2 = a1.T @ dz2                # (8,1) W2 기울기 (4샘플 합침) / W2 grad (sums the 4 samples)
    db2 = np.sum(dz2, axis=0, keepdims=True)   # (1,1) b2 기울기 / b2 grad
    da1 = dz2 @ W2.T                # (4,8) 오차를 은닉층으로 되돌림 / push error back to hidden layer
    dz1 = da1 * (z1 > 0)            # (4,8) ReLU 거꾸로 (켜진 자리만 통과) / ReLU backward (only where on)
    dW1 = X.T @ dz1                # (2,8) W1 기울기 / W1 grad
    db1 = np.sum(dz1, axis=0, keepdims=True)   # (1,8) b1 기울기 / b1 grad

    # 6. 업데이트 (기울기 반대 방향으로 lr만큼 빼기) / Update (step lr against the gradient)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # 1000번마다 손실 출력 (내려가는지 확인) / print loss every 1000 (is it dropping?)
    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}   loss {loss:.4f}")

# 7. 학습 끝 — 최종 예측 vs 정답 / Training done — final prediction vs answer
print("\n=== 학습 후 ===")
print("예측:", output.round(3).ravel())
print("정답:", y.ravel())


# ==================================================================
#  참고: 학습 전 첫 1회 순전파/역전파 스냅샷 (소수 2자리)
#  Reference: snapshot of the very first forward/backprop, before training (2 decimals)
#  ※ 기호가 헷갈릴 때 이 값들과 대조해서 보세요. (학습이 돌면 값은 바뀜)
#  ※ When a symbol is confusing, compare against these. (values change once training runs)
# ==================================================================
#  [데이터 / data]
#  X      (4,2) 입력 / input   = [[0,0],[0,1],[1,0],[1,1]]
#  y      (4,1) 정답 / answer  = [[0],[1],[1],[0]]
#
#  [가중치 (초기 랜덤) / weights (initial random)]
#  W1     (2,8) 1층 가중치 / layer-1 weights
#         = [[ 0.50 -0.14  0.65  1.52 -0.23 -0.23  1.58  0.77]
#            [-0.47  0.54 -0.46 -0.47  0.24 -1.91 -1.72 -0.56]]
#  b1     (1,8) = [0 0 0 0 0 0 0 0]
#  W2     (8,1) 2층 가중치 / layer-2 weights
#         = [-1.01, 0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42]
#  b2     (1,1) = [0]
#
#  [순전파 / forward]
#  z1     (4,8) 1층 가중합 / layer-1 weighted sum
#         = [[ 0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]   ← 샘플/sample [0,0]
#            [-0.47  0.54 -0.46 -0.47  0.24 -1.91 -1.72 -0.56]   ← 샘플/sample [0,1]
#            [ 0.50 -0.14  0.65  1.52 -0.23 -0.23  1.58  0.77]   ← 샘플/sample [1,0]
#            [ 0.03  0.40  0.18  1.06  0.01 -2.15 -0.15  0.21]]  ← 샘플/sample [1,1]
#  a1     (4,8) ReLU 후 / after ReLU
#         = [[0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00]
#            [0.00 0.54 0.00 0.00 0.24 0.00 0.00 0.00]
#            [0.50 0.00 0.65 1.52 0.00 0.00 1.58 0.77]
#            [0.03 0.40 0.18 1.06 0.01 0.00 0.00 0.21]]  ← 음수가 0으로 꺾임 / negatives clipped to 0
#  z2     (4,1) 2층 가중합 / layer-2 weighted sum = [0.00, 0.53, -4.23, -1.84]
#  output (4,1) 최종 예측 / final prediction     = [0.50, 0.63, 0.01, 0.14]   (정답/answer [0,1,1,0])
#
#  [손실 / loss]
#  loss   스칼라 / scalar = 1.3871   (예측-정답 벌점 평균, 낮을수록 좋음 / mean penalty, lower is better)
#
#  [역전파 / backprop]
#  dz2    (4,1) 오차 신호 / error signal = [0.12, -0.09, -0.25, 0.03]   = (output-y)/4
#  dW2    (8,1) W2 기울기 / W2 grad = [-0.12, -0.04, -0.15, -0.34, -0.02, 0.00, -0.39, -0.18]
#  db2    (1,1) b2 기울기 / b2 grad = [-0.18]   = dz2 4개 합 / sum of the 4 dz2 (0.12-0.09-0.25+0.03)
#  da1    (4,8) 은닉층 오차 / hidden-layer error = dz2를 W2로 뒤로 흘린 것 / dz2 pushed back via W2 (dz2 @ W2.T)
#              각 행 = dz2[그 샘플] × W2 / each row = dz2[that sample] × W2 (출력 오차 1개를 8뉴런에 배분)
#              = [[-0.13  0.04 -0.11 -0.18  0.18 -0.03  0.01 -0.18]   ← 샘플/sample [0,0]
#                 [ 0.09 -0.03  0.08  0.13 -0.14  0.02 -0.01  0.13]   ← 샘플/sample [0,1]
#                 [ 0.25 -0.08  0.22  0.35 -0.36  0.06 -0.02  0.35]   ← 샘플/sample [1,0]
#                 [-0.03  0.01 -0.03 -0.05  0.05 -0.01  0.00 -0.05]]  ← 샘플/sample [1,1]
#  dz1    (4,8) ReLU 거꾸로 / ReLU backward = da1에서 꺼진 뉴런(z1<=0) 자리를 0으로 막은 것 / zero out the off-neurons (z1<=0)
#  dW1    (2,8) W1 기울기 / W1 grad
#         = [[ 0.21  0.01  0.19  0.30  0.05  0.00 -0.02  0.30]
#            [-0.04 -0.02 -0.03 -0.05 -0.09  0.00  0.00 -0.05]]
#  db1    (1,8) b1 기울기 / b1 grad = [0.21 -0.02 0.19 0.30 -0.09 0.00 -0.02 0.30]
# ==================================================================
