# Jigsaw ACRC - 모델 결과 비교

**업데이트**: 2025-10-13 18:48

---

## 🏆 모델 성능 순위

### 1위: SetFit (Sentence Transformers) ⭐

**CV AUC**: 0.776110 (±0.014379)

**방법론**:
- Sentence Transformer 모델: `all-MiniLM-L6-v2`
- Body + Rule 임베딩 (384차원)
- Positive/Negative examples와의 cosine similarity (9개 features)
- Logistic Regression 분류기

**Feature 구성**:
- Sentence embeddings: 384 features
- Similarity features: 9 features
  - sim_pos1, sim_pos2 (Positive examples와의 유사도)
  - sim_neg1, sim_neg2 (Negative examples와의 유사도)
  - avg_pos_sim, avg_neg_sim (평균 유사도)
  - max_pos_sim, min_neg_sim (최대/최소 유사도)
  - diff_sim (positive - negative 차이)
- **총 393 features**

**학습 시간**: ~3분

**Fold별 성능**:
```
Fold 1: 0.794005
Fold 2: 0.757451
Fold 3: 0.777524
Fold 4: 0.789260
Fold 5: 0.762307

Mean: 0.776110 ± 0.014379
```

**강점**:
- ✅ Few-shot learning에 최적화
- ✅ Positive/Negative examples를 직접 활용
- ✅ 안정적인 CV (낮은 표준편차)
- ✅ 빠른 학습 속도

**약점**:
- ⚠️ 임베딩 모델의 품질에 의존
- ⚠️ Subreddit context를 명시적으로 사용하지 않음

---

### 2위: Baseline (TF-IDF + LightGBM)

**CV AUC**: 0.614210 (±0.022787)

**방법론**:
- TF-IDF vectorizer (max 10,000 features, trigrams)
- All text fields concatenated
- LightGBM classifier

**학습 시간**: ~20초

**Fold별 성능**:
```
Fold 1: 0.640534
Fold 2: 0.584903
Fold 3: 0.620461
Fold 4: 0.636079
Fold 5: 0.591233

Mean: 0.614642 ± 0.022787
```

**강점**:
- ✅ 매우 빠른 학습
- ✅ 간단하고 이해하기 쉬움
- ✅ 재현성 높음

**약점**:
- ⚠️ Few-shot examples를 단순 concatenation으로만 활용
- ⚠️ Semantic understanding 부족

---

## 📊 성능 비교

| 모델 | CV AUC | Std | 개선 | Time |
|------|--------|-----|------|------|
| **SetFit** | **0.776** | 0.014 | baseline | 3 min |
| Baseline | 0.614 | 0.023 | - | 20 sec |

**절대 개선**: +0.162 (16.2%p)
**상대 개선**: +26.4%

---

## 🎯 제출 전략

### Phase 1: 현재 (Day 1)
- ✅ Baseline 구축 및 검증
- ✅ SetFit 구축 및 검증
- 🔄 SetFit으로 첫 제출 예정

### Phase 2: 단기 (Day 2-3)
- BERT/RoBERTa fine-tuning
- Subreddit-rule risk features 추가
- Keyword features 추가
- SetFit + features hybrid model

### Phase 3: 중기 (Day 4-7)
- Ensemble: SetFit + BERT + LightGBM
- Hyperparameter tuning
- Cross-validation 전략 개선

### Phase 4: 최종 (Day 8-10)
- 최종 모델 선택
- Code Competition notebook 완성
- 제출 준비

---

## 💡 핵심 인사이트

### 1. Few-shot Learning의 효과
SetFit이 Baseline보다 26% 더 우수한 성능을 보인 것은 **Positive/Negative examples를 직접 활용**한 덕분입니다.

**EDA에서 발견한 패턴**:
- 위반 댓글이 Positive Example과 5.98% 단어 중복 (비위반 3.07%)
- **위반 댓글은 Positive Example(위반 예시)과 더 유사**

SetFit은 이 패턴을 Cosine similarity로 직접 포착하여 높은 성능을 달성했습니다.

### 2. 안정성
SetFit의 표준편차(0.014)가 Baseline(0.023)보다 낮아 **더 안정적**입니다.
→ Public/Private LB 간 격차가 적을 것으로 예상

### 3. 속도 vs 성능
- Baseline: 20초, AUC 0.614
- SetFit: 3분, AUC 0.776

**3분 투자로 26% 성능 향상** → 매우 효율적

---

## 🚀 다음 단계

### 우선순위 1: SetFit 제출
1. SetFit Kaggle Notebook 작성
2. Kaggle에 업로드
3. 첫 제출 실행 (오늘 1/5회 사용)
4. Public LB 점수 확인

### 우선순위 2: 모델 개선
1. **SetFit + Features**:
   - Subreddit-rule historical violation rate
   - Keyword scores (legal, advertising)
   - Text length features

2. **BERT fine-tuning** (선택적):
   - RoBERTa-base
   - Input: [CLS] body [SEP] rule [SEP]
   - Examples도 포함하면 token limit 주의

3. **Ensemble**:
   - SetFit (weight: 0.7)
   - BERT (weight: 0.2)
   - Baseline (weight: 0.1)

### 우선순위 3: 실험
- Larger sentence transformer: `all-mpnet-base-v2`
- Different similarity metrics: Manhattan, Euclidean
- Contrastive learning fine-tuning

---

## 📈 예상 성능

### Public Leaderboard 예상
- **보수적**: 0.70 ~ 0.75
- **낙관적**: 0.75 ~ 0.80
- **최선**: 0.80+

CV와 Public LB는 보통 약간의 차이가 있으므로, CV 0.776이 Public LB 0.72~0.78 정도로 나타날 것으로 예상합니다.

### 최종 목표
- **Public LB**: Top 20% (상위 445팀 / 2,227팀)
- **Private LB**: Top 10% (상위 223팀)
- **최종 순위**: Top 5% 도전

---

## 📝 제출 기록

### 오늘 (Day 1)
- 제출 횟수: 0/5
- 계획: SetFit v1 제출 예정

---

**다음 업데이트**: SetFit 제출 후 Public LB 점수 확인
