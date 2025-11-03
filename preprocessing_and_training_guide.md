# 딥페이크 감지 대회: 전처리 및 학습 완벽 가이드

## 📋 목차
1. [데이터 전처리 상세 가이드](#1-데이터-전처리-상세-가이드)
2. [학습 전략 상세 설명](#2-학습-전략-상세-설명)
3. [평가지표 이해](#3-평가지표-이해)
4. [실전 팁](#4-실전-팁)

---

## 1. 데이터 전처리 상세 가이드

### 1.1 전처리 파이프라인 개요

```
동영상 입력 → 프레임 추출 → 얼굴 검출 → 크롭/정렬 → 증강 → 모델 입력
```

#### **왜 전처리가 중요한가?**
- 대회에서 제공되는 **비공개 테스트셋**은 다양한 환경(조명, 해상도, 압축)으로 촬영됨
- 얼굴 외 배경 정보는 **노이즈**로 작용할 수 있음
- **일관된 얼굴 영역** 추출이 모델 일반화 성능의 핵심

---

### 1.2 동영상 → 프레임 추출 전략

#### **프레임 샘플링 방법**

##### 방법 1: 균등 샘플링 (Uniform Sampling)
```python
# 5fps로 균등하게 추출 (5초 영상 → 25프레임)
fps = 5
frame_interval = int(video_fps / fps)

for frame_count in range(total_frames):
    if frame_count % frame_interval == 0:
        # 프레임 저장
        pass
```

**장점**: 
- 동영상 전체를 고르게 커버
- 안정적인 샘플링

**단점**: 
- 중복된 장면이 많을 수 있음

##### 방법 2: 적응적 샘플링 (Adaptive Sampling)
```python
# 얼굴 크기/품질에 따라 선택적 추출
def adaptive_sampling(frames, max_frames=30):
    selected_frames = []
    
    for frame in frames:
        face_score = calculate_face_quality(frame)  # 얼굴 크기, 선명도
        if face_score > threshold:
            selected_frames.append(frame)
        
        if len(selected_frames) >= max_frames:
            break
    
    return selected_frames
```

**장점**: 
- 고품질 프레임만 선택
- 효율적인 학습

**단점**: 
- 복잡한 구현

#### **추천 설정**

| 설정 | 값 | 이유 |
|------|-----|------|
| FPS | 5 | 5초 영상 → 25프레임 (적절한 양) |
| Max Frames | 30 | 메모리 효율 + 충분한 정보 |
| 최소 해상도 | 224x224 | EfficientNet 입력 크기 |

---

### 1.3 얼굴 검출 및 크롭

#### **얼굴 검출 라이브러리 비교**

| 라이브러리 | 정확도 | 속도 | 추천 |
|-----------|--------|------|------|
| **MTCNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ 권장 (정확도 최고) |
| **RetinaFace** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ 권장 (속도+정확도) |
| **Dlib** | ⭐⭐⭐ | ⭐⭐ | ❌ 느림 |
| **OpenCV Haar** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ 부정확 |

#### **MTCNN 사용법 (권장)**

```python
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    image_size=224,       # 출력 크기
    margin=20,            # 얼굴 주변 여백 (픽셀)
    device='cuda',        # GPU 사용
    post_process=False    # 정규화 안함 (직접 처리)
)

# 얼굴 검출 + 크롭
face_tensor = mtcnn(image_rgb)  # (3, 224, 224)
```

**주요 파라미터 설명**:
- `image_size`: 최종 얼굴 이미지 크기
- `margin`: 얼굴 바운딩 박스에서 추가로 포함할 여백 (기본 20px)
- `post_process`: False로 설정 → 원본 픽셀값 유지

#### **얼굴 정렬 (Face Alignment)**

얼굴 검출 후 **눈/코/입 위치**를 기준으로 정렬하면 성능 향상:

```python
from PIL import Image
import cv2

def align_face(image, landmarks):
    """
    5개 랜드마크를 사용한 얼굴 정렬
    """
    # 눈 중심 계산
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # 회전 각도 계산
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 회전 적용
    center = tuple(np.array(image.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned
```

**정렬의 효과**:
- ✅ 모델이 일관된 얼굴 방향으로 학습
- ✅ 회전 변환에 강건해짐
- ✅ 2-3% 성능 향상 (FaceForensics++ 벤치마크)

---

### 1.4 데이터 증강 (Data Augmentation)

#### **왜 증강이 필요한가?**

대회 테스트셋은 다음과 같은 **실생활 노이즈**를 포함:
- 낮은 조명
- JPEG/H.264 압축
- 흐릿한 카메라
- 다양한 인종/연령

→ **증강으로 이러한 환경을 시뮬레이션**

#### **증강 전략**

##### 1. **기하학적 변형** (Geometric Augmentation)

```python
import albumentations as A

geometric_aug = A.Compose([
    A.HorizontalFlip(p=0.5),              # 좌우 반전
    A.ShiftScaleRotate(
        shift_limit=0.1,                   # 10% 이동
        scale_limit=0.1,                   # 10% 확대/축소
        rotate_limit=15,                   # ±15도 회전
        p=0.5
    ),
])
```

**효과**:
- 다양한 얼굴 각도 학습
- 과적합 방지

##### 2. **색상 및 밝기 조정** (Color Augmentation)

```python
color_aug = A.OneOf([
    A.RandomBrightnessContrast(
        brightness_limit=0.2,              # ±20% 밝기
        contrast_limit=0.2,                # ±20% 대비
        p=1
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,                # 색조 변경
        sat_shift_limit=30,                # 채도 변경
        val_shift_limit=20,                # 명도 변경
        p=1
    ),
], p=0.5)
```

**효과**:
- 다양한 조명 조건 대응
- 실외/실내 환경 시뮬레이션

##### 3. **압축 및 노이즈** (Compression & Noise)

```python
compression_aug = A.OneOf([
    A.ImageCompression(
        quality_lower=60,                  # JPEG 품질 60-100
        quality_upper=100,
        p=1
    ),
    A.GaussNoise(
        var_limit=(10, 50),                # 가우시안 노이즈
        p=1
    ),
    A.ISONoise(
        color_shift=(0.01, 0.05),          # ISO 노이즈 (카메라)
        intensity=(0.1, 0.5),
        p=1
    ),
], p=0.3)
```

**효과**:
- 소셜 미디어 업로드 시뮬레이션 (압축)
- 저품질 카메라 대응

##### 4. **블러** (Blur)

```python
blur_aug = A.OneOf([
    A.GaussianBlur(blur_limit=(3, 5), p=1),   # 흐림
    A.MotionBlur(blur_limit=(3, 5), p=1),     # 모션 블러
], p=0.2)
```

**효과**:
- 흔들린 영상 대응
- 실제 환경 시뮬레이션

#### **전체 증강 파이프라인**

```python
train_transform = A.Compose([
    # 1. 기하학적 변형
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    
    # 2. 색상/밝기
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
    ], p=0.5),
    
    # 3. 압축/노이즈
    A.OneOf([
        A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
        A.GaussNoise(var_limit=(10, 50), p=1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
    ], p=0.3),
    
    # 4. 블러
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.MotionBlur(blur_limit=(3, 5), p=1),
    ], p=0.2),
    
    # 5. 정규화 (ImageNet 평균/표준편차)
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**증강 적용 비율**:
- 기하학적 변형: 50%
- 색상 조정: 50%
- 압축/노이즈: 30%
- 블러: 20%

---

### 1.5 주파수 도메인 증강 (FFT)

#### **왜 주파수 도메인인가?**

딥페이크는 다음과 같은 **고주파 아티팩트**를 남김:
- 얼굴 경계의 불연속성
- 텍스처 패턴의 불일치
- 업샘플링 흔적

→ **FFT로 주파수 도메인 특징 추출**하면 성능 향상

#### **FFT 기반 증강 구현**

```python
import torch
import numpy as np

def extract_frequency_features(image):
    """
    이미지에서 주파수 도메인 특징 추출
    
    Args:
        image: (C, H, W) RGB 이미지 텐서
    
    Returns:
        freq_features: (C, H, W) 주파수 진폭 스펙트럼
    """
    # 그레이스케일 변환
    gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    
    # FFT
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    
    # 진폭 스펙트럼
    magnitude = torch.abs(fft_shift)
    log_magnitude = torch.log(magnitude + 1)  # 로그 스케일
    
    # 정규화
    log_magnitude = (log_magnitude - log_magnitude.mean()) / log_magnitude.std()
    
    return log_magnitude
```

#### **주파수 도메인 시각화**

```python
import matplotlib.pyplot as plt

def visualize_frequency(image):
    """주파수 도메인 시각화"""
    freq = extract_frequency_features(image)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 원본 이미지
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 주파수 스펙트럼
    axes[1].imshow(freq, cmap='gray')
    axes[1].set_title('Frequency Spectrum')
    axes[1].axis('off')
    
    plt.show()
```

**Real vs Fake 차이점**:
- **Real**: 자연스러운 주파수 분포
- **Fake**: 고주파 영역에 비정상적인 패턴

---

## 2. 학습 전략 상세 설명

### 2.1 데이터셋 분할 전략

#### **전체 데이터셋 구성**

```
FaceForensics++ (1,000 videos)  ─┐
Celeb-DF (5,639 videos)         ─┼─→ 병합 → 학습/검증
DFDC Preview (5,000 videos)     ─┘

WildDeepfake (707 videos)       ───→ 추론 테스트
```

#### **5-Fold Cross Validation**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Fold별로 학습
    pass
```

**장점**:
- 모든 데이터가 검증에 사용됨
- 과적합 방지
- 안정적인 성능 평가

#### **데이터 비율**

```
전체 데이터 = 100%
├─ Fold 1: Train 80% | Val 20%
├─ Fold 2: Train 80% | Val 20%
├─ Fold 3: Train 80% | Val 20%
├─ Fold 4: Train 80% | Val 20%
└─ Fold 5: Train 80% | Val 20%

최종 모델 = 5개 Fold 평균 or Best Fold 선택
```

---

### 2.2 모델 선택 가이드

#### **1. EfficientNet-B4**

**특징**:
- ImageNet 사전학습 가중치
- 효율적인 아키텍처 (Compound Scaling)
- 19M 파라미터

**장점**:
- ✅ 빠른 학습
- ✅ 적은 메모리 사용
- ✅ 좋은 일반화 성능

**단점**:
- ❌ Xception보다 약간 낮은 정확도

**추천 상황**:
- GPU 메모리 제한 (16GB 미만)
- 빠른 실험 반복

#### **2. XceptionNet**

**특징**:
- Depthwise Separable Convolution
- FaceForensics++ 벤치마크 1위

**장점**:
- ✅ 딥페이크 검출에 특화
- ✅ 주파수 아티팩트 검출 우수
- ✅ 안정적인 성능

**단점**:
- ❌ 느린 학습 속도

**추천 상황**:
- 최고 성능 목표
- 충분한 GPU 리소스

#### **3. Frequency-Aware Model (듀얼 경로)**

**특징**:
- RGB + 주파수 도메인 듀얼 경로
- 2025년 최신 연구 기법

**장점**:
- ✅ 고주파 아티팩트 검출
- ✅ 압축 공격에 강건
- ✅ SOTA 성능

**단점**:
- ❌ 높은 메모리 사용
- ❌ 복잡한 구현

**추천 상황**:
- 상위 1% 목표
- 연구 목적

---

### 2.3 손실 함수 선택

#### **1. Binary Cross Entropy (BCE)**

```python
criterion = nn.BCELoss()
```

**장점**:
- 간단한 구현
- 안정적인 학습

**단점**:
- 클래스 불균형에 약함

**추천**:
- 클래스 균형이 맞을 때 (Real:Fake = 1:1)

#### **2. Weighted BCE**

```python
# Fake가 적을 경우 pos_weight > 1
pos_weight = len(real_samples) / len(fake_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**장점**:
- 클래스 불균형 대응
- 간단한 구현

**단점**:
- 가중치 조정 필요

**추천**:
- Real:Fake ≠ 1:1 (예: 3:1)

#### **3. Focal Loss (권장)**

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

**장점**:
- ✅ 클래스 불균형 자동 대응
- ✅ 어려운 샘플에 집중 (hard negative mining)
- ✅ SOTA 성능

**단점**:
- 하이퍼파라미터 조정 필요 (alpha, gamma)

**추천**:
- 대부분의 상황 (기본 선택)

---

### 2.4 옵티마이저 및 스케줄러

#### **옵티마이저 비교**

| 옵티마이저 | 학습 속도 | 일반화 | 메모리 |
|-----------|---------|--------|--------|
| **AdamW** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **SGD + Momentum** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Adam** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

#### **추천: AdamW + Cosine Annealing**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,                # 초기 학습률
    weight_decay=1e-4       # L2 정규화
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,               # 총 에포크 수
    eta_min=1e-6            # 최소 학습률
)
```

**Cosine Annealing 효과**:
- 초기: 빠른 수렴
- 후반: 미세 조정 (fine-tuning)
- 지역 최적점 탈출

#### **학습률 스케줄 시각화**

```python
import matplotlib.pyplot as plt

lrs = []
for epoch in range(30):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Schedule')
plt.grid(True)
plt.show()
```

---

### 2.5 학습 팁

#### **1. Warmup (선택)**

초기 에포크에서 학습률을 점진적으로 증가:

```python
def warmup_lr_schedule(optimizer, epoch, warmup_epochs=5, base_lr=1e-4):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
```

**효과**:
- 초기 불안정성 방지
- 더 나은 최적점 도달

#### **2. Early Stopping**

검증 성능이 개선되지 않으면 조기 종료:

```python
best_macro_f1 = 0.0
patience = 5
patience_counter = 0

for epoch in range(epochs):
    val_loss, val_metrics = validate(...)
    
    if val_metrics['macro_f1'] > best_macro_f1:
        best_macro_f1 = val_metrics['macro_f1']
        patience_counter = 0
        # 모델 저장
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

**효과**:
- 과적합 방지
- 학습 시간 절약

#### **3. Gradient Clipping**

그래디언트 폭발 방지:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**효과**:
- 안정적인 학습
- NaN 방지

---

## 3. 평가지표 이해

### 3.1 Macro F1-score 수식

$$
\text{Macro F1} = \frac{1}{2} \left( F1_{\text{Real}} + F1_{\text{Fake}} \right)
$$

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$

### 3.2 혼동 행렬 예시

|              | Pred: Real | Pred: Fake |
|--------------|-----------|-----------|
| **True: Real** | TN = 450  | FP = 50   |
| **True: Fake** | FN = 30   | TP = 470  |

**클래스별 F1 계산**:

```
Real (Negative):
- Precision = TN / (TN + FN) = 450 / 480 = 0.9375
- Recall = TN / (TN + FP) = 450 / 500 = 0.9000
- F1_Real = 2 * (0.9375 * 0.9000) / (0.9375 + 0.9000) = 0.9184

Fake (Positive):
- Precision = TP / (TP + FP) = 470 / 520 = 0.9038
- Recall = TP / (TP + FN) = 470 / 500 = 0.9400
- F1_Fake = 2 * (0.9038 * 0.9400) / (0.9038 + 0.9400) = 0.9216

Macro F1 = (0.9184 + 0.9216) / 2 = 0.9200
```

### 3.3 왜 Macro F1인가?

- **Accuracy의 문제점**: 클래스 불균형 시 왜곡
  - 예: Real 90%, Fake 10% 데이터에서 모든 것을 Real로 예측 → Accuracy 90% (의미 없음)
- **Macro F1의 장점**: 각 클래스를 동등하게 평가
  - Real과 Fake를 모두 잘 맞춰야 높은 점수

---

## 4. 실전 팁

### 4.1 클래스 불균형 대응

#### **문제**:
실제 데이터셋은 Real:Fake ≠ 1:1 (예: 3:1)

#### **해결책**:

##### 1. **Weighted Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# 클래스별 가중치 계산
class_counts = [len(real_samples), len(fake_samples)]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

# 샘플별 가중치
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

##### 2. **SMOTE (Oversampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

##### 3. **Focal Loss** (이미 설명)

---

### 4.2 모델 앙상블

#### **앙상블 전략**

##### 1. **모델 다양성**
```python
# 서로 다른 아키텍처
model1 = EfficientNetDeepfakeDetector()
model2 = XceptionDeepfakeDetector()
model3 = FrequencyAwareDeepfakeDetector()
```

##### 2. **Soft Voting (평균)**
```python
def ensemble_predict(models, image):
    predictions = []
    for model in models:
        pred = model(image)
        predictions.append(pred.item())
    
    return np.mean(predictions)  # 확률 평균
```

##### 3. **Weighted Voting (가중 평균)**
```python
def weighted_ensemble(models, weights, image):
    predictions = []
    for model, weight in zip(models, weights):
        pred = model(image)
        predictions.append(pred.item() * weight)
    
    return sum(predictions) / sum(weights)

# 예: 검증 성능에 비례한 가중치
weights = [0.3, 0.5, 0.2]  # EfficientNet, Xception, Frequency
```

**앙상블 효과**:
- ✅ 2-5% 성능 향상
- ✅ 안정적인 예측
- ✅ 과적합 감소

---

### 4.3 Test Time Augmentation (TTA)

추론 시 증강을 적용하여 여러 예측을 평균:

```python
def tta_predict(model, image, n_tta=5):
    """
    Test Time Augmentation
    
    Args:
        model: 학습된 모델
        image: 입력 이미지
        n_tta: 증강 횟수
    
    Returns:
        avg_prediction: 평균 예측
    """
    predictions = []
    
    tta_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for _ in range(n_tta):
        augmented = tta_transform(image=image)['image']
        augmented = augmented.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(augmented)
        
        predictions.append(pred.item())
    
    return np.mean(predictions)
```

**TTA 효과**:
- ✅ 1-3% 성능 향상
- ✅ 안정적인 예측
- ⚠️ 추론 시간 증가 (n_tta배)

---

### 4.4 동영상 레벨 집계 전략

프레임별 예측 → 동영상 레벨 예측으로 집계:

#### **방법 1: Average (평균)**
```python
frame_predictions = [0.3, 0.7, 0.8, 0.6, 0.9]  # 5개 프레임
video_prediction = np.mean(frame_predictions)  # 0.66
final_label = 1 if video_prediction >= 0.5 else 0
```

**장점**:
- 부드러운 예측
- 확률 정보 유지

#### **방법 2: Majority Vote**
```python
frame_labels = [0, 1, 1, 1, 1]  # 이진화 후
video_label = 1 if sum(frame_labels) > len(frame_labels) / 2 else 0
```

**장점**:
- 명확한 결정
- 이상치에 강건

#### **방법 3: Weighted Average (가중 평균)**
```python
# 얼굴 크기/품질에 따라 가중치
face_qualities = [0.8, 0.9, 0.95, 0.7, 0.85]
weighted_pred = np.average(frame_predictions, weights=face_qualities)
```

**장점**:
- 고품질 프레임 우선
- 더 정확한 예측

**추천**: **Average (평균)** - 안정적이고 구현 간단

---

### 4.5 크로스 데이터셋 검증

학습: FaceForensics++ → 테스트: Celeb-DF

```python
# FaceForensics++로 학습
train_dataset = DeepfakeDataset('processed_data/faceforensics', ...)
model = train(train_dataset)

# Celeb-DF로 테스트 (일반화 평가)
test_dataset = DeepfakeDataset('processed_data/celebdf', ...)
test_metrics = evaluate(model, test_dataset)

print(f"Cross-dataset Macro F1: {test_metrics['macro_f1']:.4f}")
```

**목표**: Cross-dataset Macro F1 > 0.85

**효과**:
- 실제 일반화 성능 평가
- 과적합 조기 발견

---

## 5. 타임라인 및 체크리스트

### 5.1 대회 준비 타임라인

#### **Week 1: 데이터 준비**
- [ ] 데이터셋 다운로드 (FaceForensics++, Celeb-DF)
- [ ] 전처리 파이프라인 구축
- [ ] 샘플 데이터로 초기 테스트

#### **Week 2-3: 모델 학습**
- [ ] EfficientNet-B4 베이스라인
- [ ] 5-Fold CV 학습
- [ ] Xception 추가 학습
- [ ] 크로스 데이터셋 검증

#### **Week 4: 앙상블 및 최적화**
- [ ] 모델 앙상블
- [ ] TTA 적용
- [ ] 하이퍼파라미터 튜닝
- [ ] WildDeepfake로 최종 검증

#### **마감 전: 제출**
- [ ] 추론 파이프라인 최종 점검
- [ ] CSV 제출 파일 생성
- [ ] Macro F1 > 0.90 확인

---

### 5.2 최종 체크리스트

#### **전처리**
- [ ] MTCNN으로 얼굴 검출
- [ ] 224x224 크기 조정
- [ ] 얼굴 정렬 (선택)

#### **증강**
- [ ] 기하학적 변형 (Flip, Rotate)
- [ ] 색상 조정 (Brightness, Contrast)
- [ ] 압축 시뮬레이션 (JPEG)
- [ ] 노이즈 추가 (Gaussian, ISO)

#### **모델**
- [ ] EfficientNet-B4 (베이스라인)
- [ ] Xception (성능 향상)
- [ ] Frequency-Aware (SOTA)

#### **학습**
- [ ] 5-Fold Cross Validation
- [ ] Focal Loss
- [ ] AdamW + Cosine Annealing
- [ ] Early Stopping

#### **평가**
- [ ] Macro F1-score 계산
- [ ] 혼동 행렬 분석
- [ ] 크로스 데이터셋 검증

#### **추론**
- [ ] 동영상 → 프레임 추출
- [ ] 프레임별 예측
- [ ] Average 집계
- [ ] TTA (선택)
- [ ] 모델 앙상블

---

## 6. 추천 리소스

### 데이터셋
- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
- DFDC: https://www.kaggle.com/c/deepfake-detection-challenge
- WildDeepfake: https://github.com/OpenTAI/wild-deepfake

### 논문
- LNCLIP-DF (2025): Layer Normalization CLIP for Deepfake Detection
- Frequency-Aware Detection (2025): Nature Scientific Reports
- XceptionNet: FaceForensics++ Baseline

### 코드 및 라이브러리
- PyTorch: https://pytorch.org/
- Albumentations: https://albumentations.ai/
- facenet-pytorch (MTCNN): https://github.com/timesler/facenet-pytorch
- timm: https://github.com/huggingface/pytorch-image-models

---

## 📊 예상 성능

| 모델 | Macro F1 (단일) | Macro F1 (앙상블) |
|------|----------------|-------------------|
| EfficientNet-B4 | 0.88 - 0.92 | - |
| Xception | 0.90 - 0.94 | - |
| Frequency-Aware | 0.92 - 0.95 | - |
| **앙상블 (3모델)** | - | **0.94 - 0.97** |

**목표**: Macro F1 ≥ 0.95 (상위 10%)

---

## 🎯 성공의 핵심

1. **다양한 데이터셋 병합**: FaceForensics++ + Celeb-DF + DFDC
2. **강력한 증강**: 압축, 노이즈, 색상 조정
3. **5-Fold CV**: 안정적인 성능 보장
4. **모델 앙상블**: EfficientNet + Xception + Frequency
5. **크로스 검증**: WildDeepfake로 일반화 테스트

**최종 목표**: Macro F1-score 0.95+ 달성! 🏆
