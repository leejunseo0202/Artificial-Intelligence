# ResNet-50 핵심 구조
[ResNet 논문](https://arxiv.org/pdf/1512.03385)

![ResNetImage01](https://github.com/user-attachments/assets/0ff30649-e377-414a-844c-426993eaf52a)


### 1. 잔차 학습 (Residual Learning)
- **정의:** 입력값 x를 연산 결과에 그대로 더해준다(H(x) = F(x) + x).

- **효과:** 레이어가 깊더라도 역전파(BackPropagation)에서 기울기 소실(Gradient Vanishing)문제가 없다.

![ResNetImage02](https://github.com/user-attachments/assets/bc57ca9b-0afb-4355-8326-819e7763457d)


### 2. 병목 구조 (Bottleneck Design)
- **정의:** (1x1 → 3x3 → 1x1): 
1. 1x1 Conv : 채널을 줄여서 데이터를 압축시킨다. 
2. 3x3 Conv : 압축된 상태에서 특징을 추출한다. 
3. 1x1 Conv : 채널을 원래대로 크기로 복원한다.

- **효과:** 1x1 Convolution 연산을 이용하여 채널 압축해 연산량을 줄여 더 깊은 층을 쌓을 수 있게 한다.

![ResNetImage03](https://github.com/user-attachments/assets/fc70e11f-7edb-4ed8-9dbd-31f539bcf3cb)


### 3. GAP (Global Average Pooling)
Ex) [2048, 7, 7]에서 평균을 구해 -> [2048, 1, 1]로 변환


### 4. FC Layer (Classification)
Convolution/Pooling 프로세스의 결과를 통해 Classification 한다.

![ResNetImage04](https://github.com/user-attachments/assets/75bf823a-94a7-46bd-a845-1c728350644c)






# 학습 과정
## 1. 기울기 초기화
optimizer.zero_grad()

## 2. Forward Propagation (예측)
outputs = model(inputs)

## 3. Loss 계산 (정답과 비교)
loss = criterion(outputs, labels)

## 4단계: Backward (역전파)
loss.backward()

## 5단계: 가중치 업데이트
optimizer.step()