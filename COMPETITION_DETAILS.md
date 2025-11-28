# Jigsaw - Agile Community Rules Classification
## 대회 상세 정보

### 대회 개요
- **플랫폼**: Kaggle
- **URL**: https://www.kaggle.com/competitions/jigsaw-agile-community-rules
- **주최**: Jigsaw/Conversation AI
- **상금**: $100,000
- **마감일**: 2025-10-24 06:59 UTC
- **카테고리**: Featured
- **참가 팀 수**: 2,227팀 (현재)
- **태그**: Text Classification, NLP, Social Networks

### 대회 규칙
- **일일 제출 횟수**: 5회/일 ⚠️
- **최대 팀 크기**: 5명
- **팀 병합 마감**: 2025-10-16 23:59 UTC
- **신규 참가 마감**: 2025-10-16 23:59 UTC
- **제출 방식**: **Code Competition (Notebook 제출 필수)** ⚠️
- **코드 실행 환경**: Kaggle Notebooks (GPU 사용 가능)

---

## 문제 정의

### Task
Reddit 댓글(comment)이 특정 서브레딧의 규칙(rule)을 **위반하는지 예측**하는 이진 분류 문제

### Few-Shot Learning Style
각 데이터 포인트에는:
- **Positive Examples** (규칙을 위반하는 예시 2개)
- **Negative Examples** (규칙을 위반하지 않는 예시 2개)

이를 참고하여 주어진 댓글이 규칙을 위반하는지 판단해야 함

---

## 데이터 구조

### Train Dataset
- **행 수**: 2,029개
- **컬럼**: 9개

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `row_id` | int | 고유 ID |
| `body` | str | 평가할 Reddit 댓글 (예측 대상) |
| `rule` | str | 서브레딧 규칙 설명 |
| `subreddit` | str | 서브레딧 이름 |
| `positive_example_1` | str | 규칙을 위반하는 예시 1 |
| `positive_example_2` | str | 규칙을 위반하는 예시 2 |
| `negative_example_1` | str | 규칙을 위반하지 않는 예시 1 |
| `negative_example_2` | str | 규칙을 위반하지 않는 예시 2 |
| `rule_violation` | int | **Target** (0: 위반 안함, 1: 위반함) |

### Test Dataset
- **행 수**: 10개
- **컬럼**: 8개 (train에서 `rule_violation` 제외)

### Sample Submission
- **형식**: CSV
- **컬럼**: `row_id`, `rule_violation`
- **값**: 확률값 (0.0 ~ 1.0)

---

## Target 분석

### Class Distribution (Train)
- **Class 1 (위반)**: 1,031개 (50.81%)
- **Class 0 (위반 안함)**: 998개 (49.19%)
- ✅ **매우 균형잡힌 데이터셋**

---

## 데이터 특성

### Subreddits
- **고유 서브레딧 수**: 100개
- **상위 서브레딧**:
  - legaladvice: 213건
  - AskReddit: 152건
  - soccerstreams: 139건
  - personalfinance: 125건
  - relationships: 106건

### 텍스트 길이 통계
- **댓글 (body) 평균 길이**: 177자
  - 최소: 51자
  - 최대: 499자
  - 중앙값: 138자

- **규칙 (rule) 평균 길이**: 78자
  - 최소: 54자
  - 최대: 103자

---

## 평가 지표

### Metric
- **ROC-AUC (Area Under the ROC Curve)** ✅
- Metric Name: `94635_Jigsaw_Rules_AUC`
- 0.5 (random) ~ 1.0 (perfect) 범위
- 높을수록 좋음
- 이진 분류의 표준 지표

### Submission Format
```csv
row_id,rule_violation
2029,0.75
2030,0.23
2031,0.91
...
```

- `rule_violation`: 0.0 ~ 1.0 사이의 확률값
- 1에 가까울수록 규칙 위반 가능성이 높음

---

## 핵심 도전 과제

### 1. Few-Shot Learning
- Positive/Negative 예시를 효과적으로 활용하는 방법
- 예시와 평가 댓글의 유사도를 어떻게 측정할 것인가?

### 2. 다양한 Context
- 100개의 서로 다른 서브레딧
- 각 서브레딧마다 다른 규칙과 문화
- 일반화 가능한 모델 필요

### 3. Semantic Understanding
- 단순 키워드 매칭이 아닌 맥락 이해 필요
- 유사한 표현이라도 맥락에 따라 위반 여부가 다를 수 있음

### 4. Small Dataset
- Train 데이터 2,029개만 제공
- 과적합 방지 필요
- Pre-trained 모델 활용 필수

---

## 접근 방법 제안

### Baseline Approach
1. **TF-IDF + Logistic Regression**
   - body + rule + examples를 concatenate
   - 빠른 베이스라인 구축

### Advanced Approaches

#### 1. Transformer-based Models
- **BERT, RoBERTa, DistilBERT**
- Input: `[CLS] rule [SEP] body [SEP] positive_examples [SEP] negative_examples [SEP]`
- Fine-tuning on competition data

#### 2. Sentence Similarity
- Sentence-BERT (SBERT)
- 댓글과 positive/negative 예시의 유사도 계산
- 유사도를 feature로 사용

#### 3. Few-Shot Prompting
- GPT-style models
- In-context learning with examples
- Prompt engineering

#### 4. Ensemble
- 여러 모델의 예측 결합
- Cross-validation 활용

---

## 다음 단계

### 1. EDA (Exploratory Data Analysis)
- [ ] 텍스트 분석 (워드클라우드, 빈도 분석)
- [ ] 규칙별 위반 패턴 분석
- [ ] 예시의 효과 분석
- [ ] 서브레딧별 특성 분석

### 2. Baseline Model
- [ ] 간단한 TF-IDF + Logistic Regression
- [ ] 첫 제출 및 Public LB 확인

### 3. Advanced Modeling
- [ ] BERT 기반 모델 fine-tuning
- [ ] Few-shot learning 기법 실험
- [ ] Ensemble 구축

### 4. 최종 제출
- [ ] Cross-validation으로 안정성 확인
- [ ] Public/Private overfitting 방지
- [ ] 코드 정리 및 문서화

---

## 참고 자료

### 유사한 Kaggle 대회
- Jigsaw Toxic Comment Classification Challenge
- Jigsaw Unintended Bias in Toxicity Classification

### 유용한 모델/라이브러리
- **Transformers**: huggingface/transformers
- **Sentence-BERT**: sentence-transformers
- **Text Preprocessing**: nltk, spacy

### 논문
- BERT: Pre-training of Deep Bidirectional Transformers
- Few-Shot Text Classification with Pattern-Exploiting Training
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

---

**마지막 업데이트**: 2025-10-13
**데이터 다운로드 완료**: ✅
