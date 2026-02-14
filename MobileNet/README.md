# MobileNet V1, V2, V3 구조
[MobileNet V1 논문](https://arxiv.org/abs/1704.04861)  
[MobileNet V2 논문](https://arxiv.org/abs/1801.04381)  
[MobileNet V3 논문](https://arxiv.org/abs/1905.02244)  


## 용어 정리
- **swish** : 활성화 함수. f(x)=xσ(x) ,where σ(x) = 1 / (1+e^(−x))
- **NAS** (Neural Architecture Search) : 자동으로 최적의 네트워크 구조를 설계하는 기술 (레이어를 몇 층 쌓을지, 필터 크기를 얼마로 할지 등)
- **NetAdapt** : NAS가 결정한 고정된 구조 안에서 연산량의 밀도를 최적화 (각 레이어의 구체적인 출력 채널 수 등)  
​
[swish 장점](https://velog.io/@iissaacc/Swish-function)  
[NAS](https://velog.io/@ym980118/%EB%94%A5%EB%9F%AC%EB%8B%9D-Neural-Architecture-Search-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)  

  
## 1. MobileNet V1
**1. Depthwise Separable Convolution**
- **Depthwise Convolution(공간)** : 3x3x1 필터가 M개(채널수) 존재. 각 채널의 가로·세로만 연산. (기존방식 : 3x3xM 필터를 통해서 1개의 결과물이 나옴)
- **Pointwise Convolution(채널)** : 1x1xM 필터로 채널 간 연산.  

**2. Width Multiplier($\alpha$)**
- 모델의 채널 수를 조절하는 하이퍼파라미터.
- 입출력 채널 수 M,N -> αM,αN  

**3. Resolution Multiplier ($\rho$)**
- 입력 이미지의 해상도를 조절하는 하이퍼파라미터.  

![MobileNetV1Image](https://github.com/user-attachments/assets/d766af43-b361-45ee-887f-df25304b9a77)


**문제점**
1. ReLU로 인한 정보 손실
- V1은 채널 수가 적어 ReLU를 거치면 데이터의 의미 있는 특징들이 사라지는 현상 발생.  

2. Gradient Vanishing
- 단순하게 레이어를 위로만 쌓는 구조  

3. Depthwise Layer의 학습 부족
- 필터가 보는 정보량이 적어 학습해야 할 파라미터가 부족해서 충분한 특징을 추출하지 못하는 경우 발생  


## 2. MobileNet V2