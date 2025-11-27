<div align="center">

# 🔍 Kaggle Jigsaw ACRC: LLM 어댑터 호환성 디버깅

> **"메달보다 중요한 건 체계적 문제 해결 능력"**
> 한국 대학생이 5일간 투자해 배운 프로덕션 ML 디버깅 케이스 스터디

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.44-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

**🏆 최종 순위: 1,121 / 2,444팀 (상위 46%)** | **📊 점수: 0.904 ROC-AUC**

[📖 실패 분석 보기](실패분석.md) • [💻 코드 & 재현](#코드-및-재현) • [🇺🇸 English](README_EN.md)

</div>

---

## 🎯 프로젝트 핵심

**LoRA 어댑터 호환성 버그를 가설 기반 디버깅으로 분석**하고,
프로덕션 ML 시스템에 적용 가능한 검증 체크리스트를 확립한 **체계적 문제 해결 케이스 스터디**입니다.

---

## 📌 프로젝트 요약

<table>
<tr>
<td width="25%"><b>🎯 대회</b></td>
<td>Kaggle Jigsaw - 커뮤니티 규칙 위반 분류 (이진 분류)</td>
</tr>
<tr>
<td><b>⏱️ 기간</b></td>
<td>2025년 10월 20-24일 (5일, 총 20시간 투자)</td>
</tr>
<tr>
<td><b>👨‍💻 역할</b></td>
<td>개인 프로젝트 (Solo) - 대학생 개인 포트폴리오</td>
</tr>
<tr>
<td><b>🏅 최종 결과</b></td>
<td><b>1,121위 / 2,444팀 (상위 46%)</b><br>0.904 ROC-AUC (메달 실패: Bronze 0.920 필요, gap +0.016)</td>
</tr>
<tr>
<td><b>💡 핵심 성과</b></td>
<td>
✅ <b>LoRA 어댑터 호환성 버그 체계적 분석</b><br>
✅ <b>가설 기반 디버깅으로 근본 원인 규명 (80% 신뢰도)</b><br>
✅ <b>프로덕션 적용 가능한 검증 체크리스트 확립</b>
</td>
</tr>
<tr>
<td><b>🛠️ 기술 스택</b></td>
<td>DeBERTa-v3, Qwen 2.5, LoRA/PEFT, PyTorch, Transformers, Kaggle GPU</td>
</tr>
</table>

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

## 📊 대회 제출 내역

| 제출 번호 | 날짜 | 모델/방법 | Public LB | 노트북 |
|----------|------|-----------|-----------|--------|
| 1 | 2025-10-20 | DeBERTa-v3-base (Baseline v1) | ~0.900 | `kaggle_baseline_v2.ipynb` |
| 2 | 2025-10-21 | DeBERTa-v3-base (Baseline v2) | **0.904** ✅ | `kaggle_baseline_v2.ipynb` |
| 3 | 2025-10-21 | SetFit | ~0.850 | `kaggle_setfit_submission.ipynb` |
| (미제출) | 2025-10-22 | Qwen Tier 1 v1 + LoRA | - | `qwen_tier1_v2.ipynb` |
| (미제출) | 2025-10-22 | Qwen Tier 1 v2 + LoRA | - | `qwen_tier1_v2.ipynb` |

**최종 제출:** DeBERTa-v3-base Baseline v2 (0.904)
**최종 순위:** 1,121 / 2,444 (상위 46%)

---

## 🔥 실패 분석 (핵심!)

**👉 [자세한 실패 분석 과정 보기 (실패분석.md)](실패분석.md)**

### 요약: 2번의 실패와 교훈

#### 실패 1: Tier 1 v1
```python
model = Qwen 2.5 1.5B-Instruct
adapter = mahmoudmohamed/reddit-4b-think
# 결과: 모든 예측 0.0 ❌
```

#### 실패 2: Tier 1 v2
```python
# 개선: Chat template, Few-shot, Temperature 0.01
# 결과: 여전히 모든 예측 0.0 ❌ (파싱 100% 성공!)
```

**핵심 발견:** 파싱은 완벽한데 값이 전부 같다 = **프롬프트 문제가 아니다!**

### 근본 원인: 베이스 모델 불일치
- Config: "1.5B용입니다"
- 실제: 데이터셋 이름 "4b-think" (4B용)
- 결과: weight dimension 불일치 → degenerate output

### 교훈
1. **Config 파일은 거짓말할 수 있다** → 교차 검증 필수
2. **100% 성공 지표가 실패를 숨길 수 있다** → 분포 모니터링
3. **2시간 Time-boxing** → 분석 마비 방지

---

## 💬 연락처

- GitHub: [@softkleenex](https://github.com/softkleenex)
- Repository: [kaggle-jigsaw-acrc-portfolio](https://github.com/softkleenex/kaggle-jigsaw-acrc-portfolio)

---

**마지막 업데이트:** 2025년 10월 25일

**라이선스:** MIT License
