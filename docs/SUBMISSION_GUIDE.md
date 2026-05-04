# Kaggle 제출 가이드

## ⚠️ Code Competition 특징

이 대회는 **Code Competition**입니다:
- CSV 파일 직접 제출 **불가능**
- Kaggle Notebook 제출 **필수**
- API로는 Kernel 업로드만 가능, Submit은 웹 UI 필요

---

## 🚀 제출 방법

### 방법 1: API로 Kernel 업로드 후 웹 제출 (진행 중)

#### Step 1: Kernel 업로드 (완료/진행 중)
```bash
cd /mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC
export PATH=$PATH:~/.local/bin
kaggle kernels push
```

#### Step 2: 웹에서 제출 (사용자 직접)

1. **Kernel 페이지 접속**:
   https://www.kaggle.com/code/softkleenex/jigsaw-acrc-setfit-solution

2. **Submit to Competition 클릭**:
   - 우측 상단 `...` (점 3개) 메뉴 클릭
   - "Submit to Competition" 선택
   - 대회: `jigsaw-agile-community-rules` 선택
   - **Submit** 버튼 클릭

3. **실행 대기**:
   - Notebook이 Kaggle 환경에서 실행됨 (~5분)
   - 완료되면 자동으로 submission.csv 제출

4. **결과 확인**:
   - Leaderboard에서 Public Score 확인
   - 예상: 0.70 ~ 0.78

---

### 방법 2: 웹 UI로 직접 업로드 (더 간단)

#### Step 1: Notebook 생성
1. https://www.kaggle.com/code 접속
2. **New Notebook** 클릭
3. **File → Import Notebook** 선택

#### Step 2: 파일 업로드
- 파일: `kaggle_setfit_submission.ipynb` 업로드
- 경로: `/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/kaggle_setfit_submission.ipynb`

#### Step 3: 설정 확인
- Settings → Add Data
- Competition: `jigsaw-agile-community-rules` 추가
- Internet: ON
- GPU: OFF (CPU로도 충분)

#### Step 4: 제출
- 우측 상단 `...` 메뉴
- "Submit to Competition"
- Submit!

---

## ⏱️ 예상 소요 시간

| 단계 | 시간 |
|------|------|
| Kernel 업로드 | 1분 |
| 웹에서 Submit 클릭 | 30초 |
| Notebook 실행 (Kaggle) | 5분 |
| **총** | **6-7분** |

---

## 📊 예상 결과

- **CV AUC**: 0.776
- **예상 Public LB**: 0.70 ~ 0.78
- **예상 순위**: 상위 20-30% (약 450-670위 / 2,227팀)

---

## 🔍 제출 후 확인사항

### 1. Leaderboard 확인
https://www.kaggle.com/competitions/jigsaw-agile-community-rules/leaderboard

### 2. 제출 기록 확인
```bash
export PATH=$PATH:~/.local/bin
kaggle competitions submissions -c jigsaw-agile-community-rules
```

### 3. 내 순위 확인
- Kaggle 웹사이트 → Competition → Leaderboard
- 내 username 검색

---

## ❓ 문제 해결

### "Notebook not found"
→ Kernel이 아직 업로드되지 않음
→ `kaggle kernels list --mine` 으로 확인

### "Submission failed"
→ Notebook 실행 중 에러 발생
→ Notebook의 Output/Log 확인

### "Invalid submission format"
→ submission.csv 형식 오류
→ row_id, rule_violation 컬럼 확인

---

## 📝 다음 제출 준비

### 오늘 제출 횟수: 1/5 (SetFit 제출 후)

### 다음 개선 방향:
1. **Feature 추가**:
   - Subreddit-rule historical risk
   - Keyword features
   - Text length features

2. **모델 개선**:
   - BERT fine-tuning
   - Larger sentence transformer (all-mpnet-base-v2)
   - Ensemble (SetFit + BERT + Baseline)

3. **하이퍼파라미터 튜닝**:
   - Logistic Regression C 값
   - Sentence model 선택
   - CV fold 수

---

## 🎯 최종 목표

- **오늘**: SetFit 제출, Public LB 확인
- **내일**: BERT + Feature engineering
- **3-5일**: Ensemble, 최적화
- **최종**: Top 10-20% 목표

---

**업데이트**: 2025-10-13 19:00
**상태**: Kernel 업로드 진행 중
**다음 단계**: 웹 UI로 Submit to Competition
