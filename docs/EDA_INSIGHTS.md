# Jigsaw ACRC - EDA 핵심 인사이트

**분석 완료 시간**: 2025-10-13 18:39
**데이터**: train.csv (2,029 samples)

---

## 🎯 5대 핵심 인사이트

### 1. 규칙별 위반 난이도 격차가 명확하다
- **No Legal Advice**: 58.31% 위반률 (고난이도)
  - 위반 텍스트: 평균 225글자, 41단어 (길고 복잡)
  - Legal 키워드 포함률: 위반 66.61% vs 비위반 16.75%
  - 고위험 서브레딧: legaladvice(79.0%), personalfinance(72.5%)

- **No Advertising**: 43.28% 위반률 (저난이도)
  - 위반 텍스트: 평균 156글자, 21단어 (짧고 간결)
  - Advertising 키워드 포함률: 위반 53.65% vs 비위반 35.71%
  - 고위험 서브레딧: churning(90.5%), sex(89.7%)

**모델링 시사점**: 규칙별로 다른 feature weighting 필요

---

### 2. 위반 텍스트는 더 길고 정보가 많다
- **위반**: 평균 195글자, 32.6단어
- **비위반**: 평균 158글자, 23.1단어
- **차이**: +38글자(+24%), +9.5단어(+41%)

**역설적 발견**: 위반 댓글이 더 많은 설명/정보를 담고 있음
→ **내용의 의도와 맥락이 중요**, 단순 길이로 판단 불가

**모델링 시사점**:
- 텍스트 길이를 feature로 추가
- Semantic understanding 필수 (BERT/RoBERTa)

---

### 3. 서브레딧이 강력한 예측 변수다
- 서브레딧 + 규칙 조합이 **위반률 2.9%~90.5% 극단적 분포**

| 서브레딧 + 규칙 | 위반률 | 샘플 수 |
|-----------------|--------|---------|
| churning + No Advertising | 90.5% | 21개 |
| soccerstreams + No Advertising | 2.9% | 139개 |
| legaladvice + No Legal Advice | 79.0% | 210개 |

**모델링 시사점**:
- Subreddit-rule 조합의 historical violation rate를 feature로 추가
- 그러나 **test set의 unseen subreddit 대비 필요**
- Subreddit embedding 또는 rule-only fallback 전략 필수

---

### 4. 비위반은 URL과 특수문자가 많고, 위반은 자연어가 많다
- **URL 포함률**: 비위반 47.8% vs 위반 32.7% (-15.1%p)
- **대문자 비율**: 비위반 5.48% vs 위반 4.18% (-1.3%p)
- **특수문자 비율**: 비위반 8.35% vs 위반 5.60% (-2.75%p)

**해석**:
- 비위반: 단순 링크/정보 공유 (사실 나열)
- 위반: 조언/판매/요청 형태의 자연스러운 문장 (행동 유도)

**모델링 시사점**:
- URL count, special char ratio, capitalization ratio를 feature로 추가
- 문장 구조 분석 (imperative mood detection)

---

### 5. Positive/Negative Examples는 명확한 길이 패턴이 있다
- **Positive Examples**: 평균 192.7글자 (규칙 준수 예시)
- **Negative Examples**: 평균 149.1글자 (위반 예시)

**중요 발견**:
- Body와 Positive Example의 단어 중복률: 위반 5.98% vs 비위반 3.07%
- **위반이 +95% 더 유사**

**의미**:
- 위반 텍스트가 Positive Example(위반 예시)과 더 비슷함
- **규칙을 알면서도 미묘하게 위반**하는 패턴

**모델링 시사점**:
- Cosine similarity between body and examples를 핵심 feature로 사용
- SetFit, Prototypical Networks 같은 Few-shot learning 방법 유리
- Contrastive learning으로 "가까우면 위반, 멀면 안전" 학습

---

## 📌 Feature Engineering 우선순위

### High Priority (필수)
1. **Similarity Features** (SetFit)
   - `cosine_sim(body, positive_examples)` - 높을수록 위반
   - `cosine_sim(body, negative_examples)` - 낮을수록 위반
   - `diff = avg_pos_sim - avg_neg_sim` - 양수면 위반 경향

2. **Subreddit-Rule Risk Score**
   - Historical violation rate by subreddit-rule combination
   - Smoothing for unseen combinations

3. **Keyword Features**
   - Legal keywords: lawyer, attorney, sue, lawsuit, legal, court
   - Ad keywords: buy, sell, click, discount, free, check

### Medium Priority (성능 향상)
4. **Text Statistics**
   - Text length (characters, words)
   - URL count
   - Special character ratio
   - Capitalization ratio

5. **Rule-Specific Features**
   - Separate models or features for each rule

### Low Priority (실험적)
6. **Linguistic Features**
   - Sentiment score
   - Readability score
   - POS tagging (imperative mood)

---

## 🚀 모델링 전략

### Phase 1: Quick Win (현재 진행 중)
- ✅ Baseline: TF-IDF + LightGBM (CV AUC: 0.614)
- 🔄 SetFit: Sentence Similarity + LogReg (실행 중)

### Phase 2: Advanced Models
- BERT/RoBERTa fine-tuning
- DeBERTa-v3-base with examples
- Pattern-Exploiting Training (PET)

### Phase 3: Ensemble
- SetFit + BERT + LightGBM
- Weighted averaging by CV scores
- Stacking

---

## ⚠️ 주의사항

### Potential Pitfalls
1. **Subreddit Leakage**
   - Test set에 unseen subreddit 가능성
   - Subreddit feature에 과도하게 의존하면 일반화 실패

2. **Few-shot Complexity**
   - 4개 examples를 모두 활용하기 어려움 (512 token 제한)
   - Example quality가 일정하지 않을 수 있음

3. **Class Imbalance (아님)**
   - 50.8% vs 49.2%로 균형잡힘
   - 그러나 subreddit-rule 조합별로는 심한 불균형

### Validation Strategy
- **Stratified K-Fold** (기본)
- **Group K-Fold by subreddit** (일반화 테스트)
- **Leave-one-subreddit-out** (extreme case)

---

## 📊 데이터 통계 요약

```
Train Size: 2,029
- Violation: 1,031 (50.8%)
- No Violation: 998 (49.2%)

Unique Subreddits: 100
- Top 3: legaladvice (213), AskReddit (152), soccerstreams (139)

Rules: 2
- No Legal Advice: 1,017 samples (58.3% violation)
- No Advertising: 1,012 samples (43.3% violation)

Text Length:
- Body: 177 ± 114 chars
- Rule: 78 ± 25 chars
- Positive Examples: 193 ± 92 chars
- Negative Examples: 149 ± 71 chars
```

---

**다음 액션**: SetFit 결과 확인 후 성능 비교, BERT 모델 준비
