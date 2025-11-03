# 딥페이크 감지 대회 핵심 정보 요약

## 📋 대회 개요

### 기본 정보
- **대회명**: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회
- **주최**: 행정안전부, 한국지능정보사회진흥원
- **주관**: 국립과학수사연구원
- **기간**: 2025년 10월 23일 ~ 11월 20일
- **총 상금**: 9,200만원

### 목표
이미지 및 동영상 속 얼굴의 **딥페이크 여부를 판별**하는 이진 분류 모델 개발

---

## 🎯 대회 규격

### 입력 (Input)
- **형식**: 이미지 (.png, .jpg) 또는 동영상 (.mp4)
- **특징**:
  - 얼굴이 명확히 식별 가능한 인물 **1명** 포함
  - 동영상 길이: 평균 **5초** 내외
  - 다양한 인종 및 연령
  - 상용 생성 서비스 + 오픈소스 모델 + face swap/lip sync

### 출력 (Output)
- **Fake (딥페이크)**: 1
- **Real (실제)**: 0
- 동영상의 경우 **단일 예측 결과** (0 또는 1)

### 평가지표
- **주요지표**: **Macro F1-score**
- 수식:
  ```
  Macro F1 = (F1_Real + F1_Fake) / 2
  
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
- **의미**: Real과 Fake를 동등하게 평가 (클래스 불균형 대응)

---

## 📦 추천 데이터셋

### 1. FaceForensics++ (필수)
- **크기**: 1,000 원본 + 4,000+ 조작 비디오
- **조작 방법**: Deepfakes, FaceSwap, Face2Face, NeuralTextures
- **다운로드**: https://github.com/ondyari/FaceForensics
- **사용**: 학습 80%, 테스트 20%
- **특징**: 저~고품질 변형 포함, 크로스 데이터셋 테스트에 강함

### 2. Celeb-DF v2 (필수)
- **크기**: 590 원본 + 5,639 딥페이크 비디오
- **특징**: 유명인 중심, 고품질 GAN/Diffusion, 5초 클립
- **다운로드**: https://github.com/yuezunli/celeb-deepfakeforensics
- **사용**: 학습 70%, 테스트 30%
- **장점**: 상용 서비스 유사, 대회 규격에 적합

### 3. DFDC (권장)
- **크기**: 960 훈련 + 10만+ 클립 비디오
- **특징**: 배우 중심, 실생활 노이즈 포함
- **다운로드**: https://www.kaggle.com/c/deepfake-detection-challenge
- **사용**: Preview 셋 학습 80%, 테스트 20%
- **장점**: 대규모, 챌린지 벤치마크

### 4. WildDeepfake (추론용)
- **크기**: 707 인터넷 딥페이크 비디오
- **특징**: 실생활 데이터, 인터넷 유통 fake
- **다운로드**: https://github.com/OpenTAI/wild-deepfake
- **사용**: 추론 테스트 (일반화 평가)
- **장점**: 대회 실전 시뮬레이션

### 데이터셋 분할 전략
```
학습 (70%): FaceForensics++ + Celeb-DF + DFDC Preview
테스트 (15%): DeeperForensics-1.0 + DF40 (크로스 소스)
추론 (15%): WildDeepfake + ExDDV + RedFace
```

---

## 🤖 추천 모델 알고리즘

### 1. EfficientNet-B4 (베이스라인)
```python
from efficientnet_pytorch import EfficientNet

class EfficientNetDeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
```

**장점**:
- ✅ ImageNet 사전학습
- ✅ 빠른 학습 속도
- ✅ 19M 파라미터 (효율적)

**예상 성능**: Macro F1 0.88 - 0.92

### 2. XceptionNet (고성능)
```python
import timm

class XceptionDeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'xception', 
            pretrained=pretrained,
            num_classes=0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, 1),
            nn.Sigmoid()
        )
```

**장점**:
- ✅ FaceForensics++ 벤치마크 1위
- ✅ 주파수 아티팩트 검출 우수
- ✅ Depthwise Separable Convolution

**예상 성능**: Macro F1 0.90 - 0.94

### 3. Frequency-Aware Model (SOTA)
```python
class FrequencyAwareDeepfakeDetector(nn.Module):
    """
    RGB + 주파수 도메인 듀얼 경로
    """
    def __init__(self):
        super().__init__()
        # RGB 경로
        self.rgb_backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # 주파수 경로
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 융합
        self.fusion = nn.Sequential(
            nn.Linear(rgb_features + freq_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def extract_frequency(self, x):
        # FFT로 주파수 도메인 특징 추출
        fft = torch.fft.fft2(x)
        magnitude = torch.abs(torch.fft.fftshift(fft))
        return torch.log(magnitude + 1)
```

**장점**:
- ✅ 고주파 아티팩트 검출
- ✅ 압축 공격에 강건
- ✅ 2025년 최신 SOTA

**예상 성능**: Macro F1 0.92 - 0.95

### 앙상블 전략
```python
# 3개 모델 앙상블 (Soft Voting)
models = [
    EfficientNetDeepfakeDetector(),
    XceptionDeepfakeDetector(),
    FrequencyAwareDeepfakeDetector()
]

def ensemble_predict(models, image):
    predictions = []
    for model in models:
        pred = model(image)
        predictions.append(pred.item())
    return np.mean(predictions)
```

**예상 성능**: Macro F1 0.94 - 0.97

---

## 📚 핵심 라이브러리

### 필수 라이브러리
```bash
# 딥러닝 프레임워크
pip install torch torchvision torchaudio

# 모델 아키텍처
pip install timm efficientnet-pytorch

# 데이터 증강
pip install albumentations opencv-python-headless

# 얼굴 검출
pip install facenet-pytorch  # MTCNN

# 평가 및 분석
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

### 라이브러리 역할

#### 1. PyTorch
- **역할**: 딥러닝 프레임워크
- **사용**: 모델 구현, 학습, 추론

#### 2. Albumentations
- **역할**: 데이터 증강
- **사용**: 
  - 기하학적 변형 (Flip, Rotate)
  - 색상 조정 (Brightness, Contrast)
  - 압축 시뮬레이션 (JPEG)
  - 노이즈 추가 (Gaussian)

#### 3. MTCNN (facenet-pytorch)
- **역할**: 얼굴 검출 및 정렬
- **사용**: 동영상 프레임에서 얼굴 영역 추출

#### 4. timm (PyTorch Image Models)
- **역할**: 사전학습 모델 라이브러리
- **사용**: Xception, EfficientNet 등 SOTA 모델 로드

#### 5. scikit-learn
- **역할**: 머신러닝 유틸리티
- **사용**:
  - 5-Fold Cross Validation (StratifiedKFold)
  - 평가 메트릭 (f1_score, confusion_matrix)

---

## 🔧 전처리 파이프라인

### Step-by-Step 프로세스

#### Step 1: 동영상 → 프레임 추출
```python
import cv2

def extract_frames(video_path, fps=5):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames
```

#### Step 2: 얼굴 검출 및 크롭
```python
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=224, margin=20, device='cuda')

def detect_face(frame):
    face = mtcnn(frame)
    if face is not None:
        return face
    else:
        # 얼굴 검출 실패 시 중앙 크롭
        return center_crop(frame, size=224)
```

#### Step 3: 데이터 증강
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 전처리 목표
- ✅ 얼굴 영역만 추출 (배경 노이즈 제거)
- ✅ 일관된 크기 (224x224)
- ✅ 다양한 환경 시뮬레이션 (증강)

---

## 🎓 학습 전략

### 5-Fold Cross Validation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = EfficientNetDeepfakeDetector()
    
    # 학습
    for epoch in range(30):
        train_loss = train_one_epoch(model, train_loader, ...)
        val_loss, val_metrics = validate(model, val_loader, ...)
        
        # Best 모델 저장
        if val_metrics['macro_f1'] > best_f1:
            torch.save(model.state_dict(), f'best_fold{fold}.pth')
```

### 손실 함수: Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**장점**: 클래스 불균형 자동 대응, 어려운 샘플에 집중

### 옵티마이저: AdamW + Cosine Annealing
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
```

**장점**: 안정적인 수렴, 후반부 미세 조정

---

## 🔍 추론 파이프라인

### 동영상 레벨 예측
```python
class DeepfakeInference:
    def predict_video(self, video_path):
        # 1. 프레임 추출
        frames = extract_frames(video_path)
        
        # 2. 얼굴 검출
        faces = [detect_face(frame) for frame in frames]
        
        # 3. 프레임별 예측
        frame_predictions = []
        for face in faces:
            pred = model(face)
            frame_predictions.append(pred.item())
        
        # 4. 집계 (Average)
        video_prediction = np.mean(frame_predictions)
        
        # 5. 이진화
        final_label = 1 if video_prediction >= 0.5 else 0
        
        return final_label, video_prediction
```

### Test Time Augmentation (TTA)
```python
def tta_predict(model, image, n_tta=5):
    predictions = []
    for _ in range(n_tta):
        augmented = tta_transform(image=image)['image']
        pred = model(augmented.unsqueeze(0))
        predictions.append(pred.item())
    return np.mean(predictions)
```

**효과**: 1-3% 성능 향상

---

## 📊 평가지표 상세

### Macro F1-score 계산 예시

#### 혼동 행렬
|              | Pred: Real | Pred: Fake |
|--------------|-----------|-----------|
| **True: Real** | TN = 450  | FP = 50   |
| **True: Fake** | FN = 30   | TP = 470  |

#### F1 계산
```python
# Real (Negative)
precision_real = TN / (TN + FN) = 450 / 480 = 0.9375
recall_real = TN / (TN + FP) = 450 / 500 = 0.9000
f1_real = 2 * (0.9375 * 0.9000) / (0.9375 + 0.9000) = 0.9184

# Fake (Positive)
precision_fake = TP / (TP + FP) = 470 / 520 = 0.9038
recall_fake = TP / (TP + FN) = 470 / 500 = 0.9400
f1_fake = 2 * (0.9038 * 0.9400) / (0.9038 + 0.9400) = 0.9216

# Macro F1
macro_f1 = (0.9184 + 0.9216) / 2 = 0.9200
```

### 목표 성능
- **기본**: Macro F1 ≥ 0.90
- **상위 10%**: Macro F1 ≥ 0.95
- **리더보드 1위**: Macro F1 ≥ 0.97

---

## 🚀 성능 향상 체크리스트

### 데이터
- [ ] FaceForensics++ + Celeb-DF 병합
- [ ] DFDC Preview 추가 학습
- [ ] WildDeepfake 추론 테스트

### 전처리
- [ ] MTCNN 얼굴 검출
- [ ] 224x224 크기 조정
- [ ] 데이터 증강 (압축, 노이즈, 색상)

### 모델
- [ ] EfficientNet-B4 베이스라인
- [ ] Xception 고성능 모델
- [ ] Frequency-Aware SOTA 모델
- [ ] 3개 모델 앙상블

### 학습
- [ ] 5-Fold Cross Validation
- [ ] Focal Loss
- [ ] AdamW + Cosine Annealing
- [ ] Early Stopping

### 추론
- [ ] 프레임별 예측 → Average 집계
- [ ] Test Time Augmentation
- [ ] 모델 앙상블

### 검증
- [ ] Macro F1-score 계산
- [ ] 혼동 행렬 분석
- [ ] 크로스 데이터셋 검증

---

## 📈 예상 성능 로드맵

| 단계 | 구현 | 예상 Macro F1 |
|------|------|---------------|
| 1 | EfficientNet-B4 베이스라인 | 0.88 - 0.90 |
| 2 | + 데이터 증강 (압축, 노이즈) | 0.90 - 0.92 |
| 3 | + Xception 모델 | 0.92 - 0.94 |
| 4 | + 5-Fold CV | 0.93 - 0.95 |
| 5 | + 모델 앙상블 (3개) | 0.94 - 0.96 |
| 6 | + TTA | 0.95 - 0.97 |

---

## 🎯 최종 전략 요약

### 핵심 3가지
1. **다양한 데이터셋 병합**
   - FaceForensics++ (기본)
   - Celeb-DF (고품질)
   - DFDC (대규모)
   - WildDeepfake (실전)

2. **강력한 모델 앙상블**
   - EfficientNet-B4 (효율성)
   - Xception (정확도)
   - Frequency-Aware (SOTA)

3. **철저한 검증**
   - 5-Fold Cross Validation
   - 크로스 데이터셋 테스트
   - WildDeepfake 일반화 평가

### 시간 배분
- Week 1: 데이터 준비 (20%)
- Week 2-3: 모델 학습 (50%)
- Week 4: 앙상블 및 최적화 (30%)

### 최종 목표
**Macro F1-score 0.95+ 달성으로 상위 10% 진입!** 🏆

---

## 📚 참고 자료

### 공식 링크
- 대회 페이지: https://aifactory.space/task/9197/overview
- 데이터 규격: https://aifactory.space/task/9197/data

### 데이터셋
- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
- DFDC: https://www.kaggle.com/c/deepfake-detection-challenge
- WildDeepfake: https://github.com/OpenTAI/wild-deepfake

### 최신 논문 (2025)
- LNCLIP-DF: https://arxiv.org/abs/2508.06248
- Frequency-Aware Detection: Nature Scientific Reports
- Comprehensive Evaluation: Applied Sciences

### 라이브러리 문서
- PyTorch: https://pytorch.org/docs/
- Albumentations: https://albumentations.ai/docs/
- facenet-pytorch: https://github.com/timesler/facenet-pytorch
- timm: https://huggingface.co/docs/timm/

---

## 💡 추가 팁

### GPU 메모리 최적화
```python
# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 배치 크기 가이드
| GPU 메모리 | EfficientNet-B4 | Xception | Frequency-Aware |
|-----------|----------------|----------|-----------------|
| 8GB | 16 | 8 | 8 |
| 16GB | 32 | 16 | 16 |
| 24GB | 64 | 32 | 24 |
| 48GB | 128 | 64 | 48 |

### 학습 시간 예상
- EfficientNet-B4: 30 epochs × 5 folds = **8-12시간** (RTX 3090)
- Xception: 30 epochs × 5 folds = **12-18시간** (RTX 3090)
- Frequency-Aware: 30 epochs × 5 folds = **15-20시간** (RTX 3090)

**총 학습 시간**: 약 35-50시간 (3-4일)

---

**행운을 빕니다! 최선을 다해 좋은 결과 얻으시길 바랍니다! 🚀**
