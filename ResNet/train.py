import torch
import torch.nn as nn
import torch.optim as optim
from model import resnet50

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = resnet50(num_classes=1000).to(device)

# 2. 손실함수 및 최적화 도구
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습
def train(model, train_loader, epochs=10):
    model.train() # 학습 모드로 설정

    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. 기울기 초기화
            optimizer.zero_grad()

            # 2. Forward Propagation (예측)
            outputs = model(inputs)

            # 3. Loss 계산 (정답과 비교)
            loss = criterion(outputs, labels)

            # 4단계: Backward (역전파)
            loss.backward()

            # 5단계: 가중치 업데이트
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("학습 완료!")

if __name__ == "__main__":
    # 1. 이미지 전처리 (이미지 크기를 224x224로 변환)
    transform = transforms.Compose([
        transforms.Resize(224), # 32x32 -> 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 데이터셋 다운로드 및 로드
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # 3. 데이터 로더 (배치 크기 설정)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

    # 4. 함수 호출 시 인자 전달
    train(model, train_loader, epochs=5)
