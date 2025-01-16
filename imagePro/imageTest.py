import os
import random
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from xgboost.dask import predict

from DjangoPro.settings import STATIC_DIR


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # conv1: 입력(3, 128, 128)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # conv2: 입력(8, 128, 128)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # conv3: 입력(16, 64, 64)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # conv4: 입력(32, 32, 32)
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # conv5: 입력(64, 16, 16)
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # maxpooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # fully connected layer. 128 설정 이유
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # 5개의 클래스로 분류

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)  # (8, 64, 64)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)  # (16, 32, 32)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)  # (32, 16, 16)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool(x)  # (64, 8, 8)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.pool(x)  # (128, 4, 4)

        # 특징 추출
        x = x.view(-1, 128 * 4 * 4)  # 128*4*4로 flatten. -1 -> batch size: 100
        # 분류
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)  # softmax를 통해 확률값으로 변환
        return x

def data_process(path): # 데이터 전처리
    torch.manual_seed(777)
    # image size: 128*128
    IMAGE_SIZE = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #GPU 사용 가능 여부에 따라 device 설정
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    data_path = os.path.join(STATIC_DIR, "test_image/") # 이미지 경로
    test_data=ImageFolder(data_path, transform=transforms.Compose([ # 이미지 폴더에서 이미지를 불러옴
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 이미지 사이즈 조정
        transforms.ToTensor() # 이미지를 텐서로 변환 -> 0~1 사이의 값으로 변환
    ]))

    test_loader = DataLoader(test_data, batch_size=10, shuffle=False) # 데이터 로더 생성
    test_images, labels = next(iter(test_loader)) # 이미지와 라벨을 가져옴
    print(test_images.shape) # 이미지 shape 출력
    print(labels)

    model2 = CNN().to(device) # CNN 모델 생성
    model_path=os.path.join(STATIC_DIR, "data/model.pt") # 모델 경로
    model2.load_state_dict(torch.load(model_path)) # 모델 불러오기
    model2.eval() # 모델 평가 모드로 설정

    test_images = test_images.to(torch.float32) # 이미지를 float32로 변환
    predict=model2(test_images.to(device)).argmax(dim=1) # 이미지를 모델에 넣어 예측
    print(predict) # 예측값 출력
    return predict








