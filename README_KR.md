# Kaggle Jigsaw ACRC: 체계적 디버깅으로 배우는 LLM 어댑터 호환성 분석

> **"실패를 통해 배우는 프로덕션 ML 디버깅 능력 증명"**

[🇺🇸 English Version](README.md) | **🇰🇷 한국어 버전**

---

## 🎯 프로젝트 한 줄 정의

Kaggle LLM 대회에서 발생한 **LoRA 어댑터 호환성 문제**를 가설 기반 디버깅으로 분석하고, 프로덕션 ML 시스템에서 적용 가능한 교훈을 도출한 **체계적 문제 해결 케이스 스터디**입니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| **대회** | [Kaggle - Jigsaw 커뮤니티 규칙 분류](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) |
| **참가 기간** | 2024년 10월 20일 ~ 24일 (5일간) |
| **투자 시간** | 약 20시간 |
| **최종 순위** | 1,121위 / 2,444팀 (상위 46%) |
| **최종 점수** | 0.904 ROC-AUC (DeBERTa 베이스라인) |
| **메달 획득** | ❌ 실패 (목표: 0.920+ Bronze) |
| **기술 스택** | Python, PyTorch, Transformers, PEFT/LoRA, Qwen 2.5 1.5B |
| **핵심 성과** | ✅ **2번의 실패를 체계적으로 분석**<br>✅ **가설 기반 근본 원인 규명 (80% 신뢰도)**<br>✅ **프로덕션 적용 가능한 검증 체크리스트 확립** |

---

## 🔑 핵심 결과 요약

### 문제 현상
Qwen 2.5 1.5B-Instruct 모델에 공개 LoRA 어댑터 적용 시, **모든 예측값이 0.0으로 출력**되는 문제 발생
- Tier 1 v1: 기본 프롬프트 → 결과: 모두 0.0 (파싱 성공률 50%)
- Tier 1 v2: 초구조화 프롬프트 → 결과: 여전히 모두 0.0 (파싱 성공률 **100%**)

### 분석 과정
**2시간 Time-boxing** 내에서 3가지 가설을 수립하고 체계적인 증거 수집을 통해 원인 추적:

| 가설 | 신뢰도 | 핵심 증거 |
|------|--------|-----------|
| **A. 베이스 모델 불일치** | **80%** | 데이터셋 이름 "4b-think" ↔ Config "1.5B" 충돌 |
| B. 이진 분류 학습 | 60% | 0.0만 출력 (경계값 패턴) |
| C. 프롬프트 형식 불일치 | 40% | 100% 파싱 성공이 반증 |

### 결론
**근본 원인:** 4B 모델용 어댑터를 1.5B 모델에 로드하여 weight dimension 불일치 발생
- PEFT 라이브러리가 오류 없이 로드하지만 (graceful degradation)
- 실제로는 weights가 제대로 align되지 않아 degenerate output 발생

### 실무적 교훈 (Production Takeaway)
**"Config 파일은 거짓말할 수 있다"**
- ✅ 설정 파일만 믿지 말고 **데이터셋 메타데이터, 학습 아티팩트, 소규모 테스트** 교차 검증 필수
- ✅ 100% 파싱 성공 같은 단일 지표가 아닌 **출력 분포 모니터링** 필요
- ✅ 외부 모델/어댑터 통합 시 **자동화된 호환성 검증 파이프라인** 구축 필요

---

## 💡 왜 이 프로젝트가 중요한가?

### 대부분의 포트폴리오
> "GPT 모델로 정확도 95% 달성! 🎉"

### 이 프로젝트
> "2번 실패했지만, **왜** 실패했는지 **어떻게** 알아냈는지 **무엇을** 배웠는지 체계적으로 설명합니다."

**실무에서 정말 필요한 능력:**
- ✅ 새로운 라이브러리/모델이 작동 안 할 때 **체계적 디버깅 능력**
- ✅ 제한된 시간에 **우선순위 결정** (2시간 time-box)
- ✅ **증거 기반 의사결정** (감이 아닌 데이터로 가설 검증)
- ✅ 복잡한 시스템에서 **근본 원인 규명** (표면적 증상이 아닌 본질 파악)

**이 포트폴리오는 이런 능력들을 실제 사례로 증명합니다.**

성공한 모델보다, **예측 불가능한 버그를 해결하는 과정**에서 엔지니어의 진짜 역량이 드러납니다. 이 프로젝트는 화려한 성능 지표 대신, 복잡한 문제에 부딪혔을 때 제가 **어떤 방식으로 사고하고, 가설을 세우고, 증거를 찾아 문제를 해결하는지**를 보여주는 생생한 기록입니다.

---

## 🛠️ 기술 스택

### ML/DL 프레임워크
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.44-FFD21E?style=flat)

### 모델
![DeBERTa](https://img.shields.io/badge/DeBERTa--v3-140M_params-blue?style=flat)
![Qwen](https://img.shields.io/badge/Qwen_2.5-1.5B_Instruct-red?style=flat)
![LoRA](https://img.shields.io/badge/LoRA-PEFT-green?style=flat)

### 플랫폼
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📖 목차

1. [프로젝트 배경](#프로젝트-배경)
2. [시도한 접근 방법](#시도한-접근-방법)
3. [🔥 실패 분석 (핵심!)](#-실패-분석-핵심)
4. [기술적 상세](#기술적-상세)
5. [배운 점과 개선 방향](#배운-점과-개선-방향)
6. [코드 및 재현](#코드-및-재현)
7. [참고 자료](#참고-자료)

---

## 프로젝트 배경

### 대회 설명
Kaggle의 "Jigsaw - Agile Community Rules Classification" 대회는 Reddit 커뮤니티의 규칙 위반 여부를 판단하는 이진 분류 문제입니다.

**입력:**
- 커뮤니티 규칙 (rule)
- 평가할 게시글 (body)
- 위반 예시 2개 (positive_example_1/2)
- 비위반 예시 2개 (negative_example_1/2)

**출력:**
- 규칙 위반 확률 (0.0 ~ 1.0)

**평가 지표:** ROC-AUC

### 경쟁 현황
- **참가 팀:** 2,444팀
- **메달 기준 (추정):**
  - 🥇 Gold: 0.933+ (Top 3)
  - 🥈 Silver: 0.925+ (Top 10)
  - 🥉 Bronze: 0.920+ (Top 40)
- **내 점수:** 0.904 (gap: +0.016 to Bronze)

---

## 시도한 접근 방법

### 타임라인

```
📅 10월 20-21일: DeBERTa 베이스라인
├─ DeBERTa-v3-base (140M params) 파인튜닝
├─ 결과: 0.904 ROC-AUC ✅
└─ 안정적이지만 메달권에는 부족

📅 10월 21일: 공개 LoRA 어댑터 발견
├─ Discussion 포럼 분석으로 Qwen 모델 공개 어댑터 발견
├─ mahmoudmohamed, seojinpark 등 15+ 어댑터 확인
└─ 전략: 학습 없이 추론만으로 빠른 시도

📅 10월 22일 오전: Tier 1 v1 실패
├─ Qwen 2.5 1.5B + mahmoudmohamed 어댑터
├─ 기본 프롬프트
└─ 결과: 모든 예측 0.0 ❌

📅 10월 22일 오후: Tier 1 v2 실패
├─ 초구조화 프롬프트 (chat template, few-shot)
├─ 3단계 파싱 전략
└─ 결과: 여전히 모든 예측 0.0 ❌ (하지만 파싱 100% 성공!)

📅 10월 22-24일: 근본 원인 분석
├─ 가설 수립 및 증거 수집
├─ seojinpark 어댑터 다운로드 (검증용)
└─ 대회 종료 (Tier 1 v3 미실행)
```

---

## 🔥 실패 분석 (핵심!)

**👉 [자세한 실패 분석 과정 보기 (실패분석.md)](실패분석.md)**

이 섹션이 이 포트폴리오의 **가장 중요한 부분**입니다.

### 요약: 2번의 실패와 교훈

#### 실패 1: Tier 1 v1
```python
# 시도 내용
model = Qwen 2.5 1.5B-Instruct
adapter = mahmoudmohamed/reddit-4b-think
prompt = "You are a content moderator..."

# 결과
predictions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parsing_success_rate = 50%
```

**가설:** "프롬프트가 문제인가?"

#### 실패 2: Tier 1 v2
```python
# 개선 사항
✅ Chat template (system/user 분리)
✅ Few-shot examples (0.95, 0.23, 0.78...)
✅ "ONLY NUMBER" 명시
✅ Temperature 0.01 (deterministic)
✅ 3단계 robust parsing

# 결과
predictions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
parsing_success_rate = 100% ✅
```

**핵심 발견:**
> 파싱은 완벽한데 값이 전부 같다 = **프롬프트 문제가 아니다!**

### 근본 원인: 베이스 모델 불일치

**증거 1: Config vs 데이터셋 이름 충돌**
```json
// mahmoudmohamed adapter config
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"  ✅
}

// 하지만 데이터셋 이름
"mahmoudmohamed/reddit-4b-think"  🚩 "4b"!
```

**증거 2: 학습 아티팩트 부재**
```
mahmoudmohamed/
├── adapter_model.bin ✅
├── adapter_config.json ✅
└── train.pkl ❌ 없음!
```

**증거 3: seojinpark 어댑터와 비교**
```json
// seojinpark fold3 config
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
  // 명시적으로 GPTQ-Int4까지 정확히 명시 ✅
}

// 파일 구조
seojinpark/fold3/
├── adapter_model.bin ✅
├── adapter_config.json ✅
├── train.pkl (5.9MB) ✅
├── val.pkl (726KB) ✅
└── fold0~fold4 전부 공개 ✅
```

### 교훈

**1. Config 파일은 거짓말할 수 있다**
- Config: "1.5B용입니다"
- 실제: 데이터셋 이름 "4b-think" (4B용)
- 해결책: **교차 검증** (config + 데이터셋 메타데이터 + 학습 아티팩트 + 소규모 테스트)

**2. 100% 성공 지표가 실패를 숨길 수 있다**
- 파싱 성공률: 100% ✅
- 하지만 모든 값이 0.0 (분포 붕괴)
- 해결책: **분포 모니터링** (mean, std, min, max)

**3. 2시간 Time-boxing으로 분석 마비 방지**
- v1: 2시간 → v2: 2시간
- 무한정 디버깅 방지
- 명확한 pivot 기준 확립

---

## 기술적 상세

### DeBERTa 베이스라인

**모델:** `microsoft/deberta-v3-base` (140M params)
- Disentangled attention mechanism
- Few-shot learning에 강점

**결과:** 0.904 ROC-AUC
- 안정적이고 재현 가능
- 메달권에는 부족하지만 solid baseline

### Qwen LoRA 추론 파이프라인

**아키텍처:**
```
Qwen 2.5 1.5B-Instruct (base model)
    ↓
+ LoRA Adapter (r=16, lora_alpha=32)
    ↓
Inference (추론만, 학습 X)
    ↓
Output: 0.0~1.0 probability
```

**Why inference-only?**
- ❌ NumPy 2.x 호환성 이슈로 학습 라이브러리 사용 불가
- ✅ 공개 어댑터 활용으로 빠른 시도 (2시간/시도)

### 프롬프트 엔지니어링 진화

**v1 → v2 개선 사항:**

| Aspect | v1 | v2 |
|--------|----|----|
| Format | Plain text | Chat template (system/user) |
| Instructions | 모호함 | "ONLY NUMBER" 명시 |
| Examples | 없음 | Few-shot (0.95, 0.23...) |
| Temperature | 0.1 | 0.01 (더 deterministic) |
| Max tokens | 50 | 10 (간결한 출력 강제) |
| Parsing | 단순 float() | 3단계 robust |

**결과:** 파싱 50% → 100% 개선 (하지만 값은 여전히 0.0)

---

## 배운 점과 개선 방향

### 프로덕션 환경에 적용 가능한 교훈

#### 1. 외부 모델 통합 시 검증 체크리스트
```python
def validate_external_adapter(adapter_path):
    """프로덕션 환경에서 외부 어댑터 사용 전 검증"""
    checks = {
        # 1. Config 검증
        "config_matches": verify_config_base_model(),

        # 2. 메타데이터 일관성
        "metadata_consistent": check_dataset_name_consistency(),

        # 3. 학습 아티팩트 존재
        "has_training_artifacts": os.path.exists("train.pkl"),

        # 4. 소규모 테스트
        "small_test_passes": test_on_3_samples(),

        # 5. 분포 정상성
        "distribution_ok": check_output_distribution()
    }

    if not all(checks.values()):
        raise ValidationError(f"Failed checks: {checks}")
```

#### 2. 모니터링 지표 확장
```python
# ❌ 단일 지표만 보지 말기
assert parsing_success_rate == 1.0  # 이것만으로 충분하지 않음

# ✅ 분포 모니터링
predictions = model.predict(test_set)
assert np.std(predictions) > 0.05, "Too uniform!"
assert np.max(predictions) > 0.3, "No high scores!"
assert 0.2 < np.mean(predictions) < 0.8, "Skewed distribution!"
```

#### 3. CI/CD 파이프라인에 설정 검증 추가
만약 이 문제가 CI/CD 파이프라인의 자동 모델 배포 단계에서 발생했다면, 모든 실시간 예측 API가 다운되었을 것입니다.

**해결책:**
- 모델 아티팩트뿐만 아니라 모든 설정 파일 (`config.json`, `adapter_config.json`) 함께 버전 관리
- 배포 전 설정 값의 유효성을 검증하는 자동화된 테스트 단계 필수

### 다음에 다르게 할 것

**만약 2시간이 더 있었다면:**
```python
# seojinpark fold3 어댑터로 즉시 테스트
model = load_qwen_1_5b_gptq()  # 정확한 variant 매칭
adapter = load_adapter("seojinpark/fold3")

predictions = predict(test_samples)
# 예상: [0.23, 0.87, 0.05, 0.91, ...] 다양한 값 출력
# → 가설 A를 100% 확인 가능
```

**만약 2일이 더 있었다면:**
1. ✅ 15+ 공개 어댑터 전수 조사 및 호환성 matrix 작성
2. ✅ Multi-fold ensemble (seojinpark fold0~4)
3. ✅ 예상 LB: 0.92-0.925 (Bronze medal 권!)

---

## 코드 및 재현

### Repository 구조
```
Jigsaw-ACRC/
├── README_KR.md (이 파일)
├── 실패분석.md (상세 분석)
├── notebooks/
│   └── tier1_v2_ultra_structured/
│       ├── qwen_tier1_v2.ipynb
│       ├── kernel-metadata.json
│       └── tier1_v2_output/
│           └── submission.csv (전부 0.0인 결과)
├── configs/
│   ├── mahmoudmohamed_adapter_config.json
│   └── seojinpark_fold3_adapter_config.json
└── docs/
    ├── RESEARCH_PROCESS_KR.md (공개 어댑터 찾은 과정)
    └── ADAPTER_COMPARISON_KR.md (어댑터 비교 분석)
```

### 핵심 코드

**Tier 1 v2 프롬프트:**
```python
def create_prompt_v2(tokenizer, row):
    system = "당신은 정확한 AI입니다. 0.0과 1.0 사이의 숫자만 응답하세요."

    user = f"""이 게시글이 규칙을 위반하는지 분석하세요.

규칙: {row['rule']}
게시글: {row['body']}

숫자만 응답하세요 (0.0~1.0):
예시: 0.95, 0.23, 0.78, 0.02

답변 (숫자만):"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

### 환경 설정
```bash
# 의존성 설치
pip install transformers==4.44.2 peft==0.12.0 torch==2.4.0

# 노트북 실행
cd notebooks/tier1_v2_ultra_structured/
jupyter notebook qwen_tier1_v2.ipynb
```

**주의:** GPU 12GB+ 필요 (FP16 추론)

---

## 참고 자료

### 프로젝트 문서
- **[실패분석.md](실패분석.md)** - 상세한 디버깅 과정 (필독!)
- **[TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)** - 기술 상세 (영어)
- **[notebooks/README_KR.md](notebooks/README_KR.md)** - 코드 설명

### 외부 링크
- [Kaggle 대회 페이지](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- [Qwen 2.5 모델 카드](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [PEFT 라이브러리 문서](https://huggingface.co/docs/peft)

---

## 💬 피드백 및 연락

이 프로젝트가 도움이 되셨다면:
- ⭐ GitHub Star를 눌러주세요!
- 💬 Issue나 PR로 피드백 주시면 감사하겠습니다

**연락처:**
- GitHub: [@softkleenex](https://github.com/softkleenex)
- Repository: [kaggle-jigsaw-acrc-portfolio](https://github.com/softkleenex/kaggle-jigsaw-acrc-portfolio)

---

**마지막 업데이트:** 2024년 10월 25일

**라이선스:** MIT License
