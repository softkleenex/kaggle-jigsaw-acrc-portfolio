# GitHub 업로드 가이드

포트폴리오를 GitHub에 업로드하는 3가지 방법을 안내합니다.

---

## 🚀 방법 1: Python API 스크립트 (가장 빠름, 추천!)

### 1단계: GitHub Personal Access Token 생성

1. https://github.com/settings/tokens/new 방문
2. 설정:
   - **Note:** `Portfolio Upload`
   - **Expiration:** `30 days` (원하는 기간)
   - **Scopes:** ✅ **repo** (전체 선택)
3. `Generate token` 클릭
4. 생성된 token 복사 (⚠️ 다시 볼 수 없으니 복사 필수!)

### 2단계: 환경변수 설정 및 실행

**Linux/Mac/WSL:**
```bash
cd /mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC

# Token 설정
export GITHUB_TOKEN="ghp_your_token_here"

# 스크립트 실행
python3 github_upload.py
```

**Windows PowerShell:**
```powershell
cd C:\LSJ\dacon\dacon\Jigsaw-ACRC

# Token 설정
$env:GITHUB_TOKEN="ghp_your_token_here"

# 스크립트 실행
python github_upload.py
```

### 3단계: 프롬프트 따라 입력

```
GitHub username 입력: YOUR_USERNAME
```

스크립트가 자동으로:
- ✅ Repository 생성
- ✅ 모든 파일 업로드
- ✅ 완료 메시지 출력

**완료!** 🎉

---

## 🖥️ 방법 2: GitHub Desktop (UI 선호 시)

### 1단계: GitHub Desktop 설치

- https://desktop.github.com/ 다운로드 및 설치
- GitHub 계정으로 로그인

### 2단계: Repository 추가

1. GitHub Desktop 열기
2. `File` → `Add Local Repository`
3. 폴더 선택: `C:\LSJ\dacon\dacon\Jigsaw-ACRC`
4. `Add Repository` 클릭

### 3단계: Publish

1. `Publish repository` 버튼 클릭
2. 설정:
   - **Name:** `kaggle-jigsaw-acrc-portfolio`
   - **Description:** `Systematic debugging of LLM adapter compatibility`
   - ✅ **Keep this code private** 체크 해제 (Public)
3. `Publish Repository` 클릭

**완료!** 🎉

---

## 💻 방법 3: Git 명령어 (수동)

### 1단계: Git 초기화

```bash
cd /mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC

git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2단계: 파일 추가 및 Commit

```bash
git add README.md
git add FAILURE_ANALYSIS.md
git add TECHNICAL_DEEP_DIVE.md
git add PORTFOLIO_REQUIREMENTS.txt
git add GITHUB_SETUP.md
git add LICENSE
git add .gitignore
git add notebooks/
git add configs/
git add docs/

git commit -m "Initial portfolio commit - Jigsaw ACRC systematic debugging case study"
```

### 3단계: GitHub Repository 생성

1. https://github.com/new 방문
2. 설정:
   - **Repository name:** `kaggle-jigsaw-acrc-portfolio`
   - **Description:** `Systematic debugging of LLM adapter compatibility`
   - **Visibility:** Public
   - ❌ Initialize with README (체크 해제)
3. `Create repository` 클릭

### 4단계: Push

```bash
# YOUR_USERNAME을 본인 GitHub username으로 변경
git remote add origin https://github.com/YOUR_USERNAME/kaggle-jigsaw-acrc-portfolio.git
git branch -M main
git push -u origin main
```

GitHub username과 token(password) 입력 요청 시:
- **Username:** YOUR_USERNAME
- **Password:** Personal Access Token (방법 1의 1단계 참고)

**완료!** 🎉

---

## 🔧 문제 해결

### "Permission denied" 오류

**원인:** Git 권한 설정 문제

**해결:**
```bash
# WSL에서
git config --global core.filemode false
```

### "Authentication failed" 오류

**원인:** Token이 잘못되었거나 권한 부족

**해결:**
1. Token 재생성 (repo 권한 확인)
2. Token 재설정: `export GITHUB_TOKEN="new_token"`

### "Repository already exists" 오류

**원인:** 이미 같은 이름의 repository 존재

**해결:**
1. 기존 repository 삭제, 또는
2. 다른 이름 사용: `kaggle-jigsaw-acrc-portfolio-v2`

### Python 스크립트 실행 시 "requests module not found"

**해결:**
```bash
pip install requests
```

---

## ✅ 업로드 확인 사항

업로드 완료 후 GitHub에서 확인:

1. **README.md 렌더링**
   - Executive Summary 잘 보이는지
   - Navigation links 작동하는지

2. **FAILURE_ANALYSIS.md**
   - 가장 중요한 파일!
   - 표와 코드 블록 잘 보이는지

3. **파일 구조**
   ```
   ✅ README.md
   ✅ FAILURE_ANALYSIS.md
   ✅ TECHNICAL_DEEP_DIVE.md
   ✅ notebooks/
   ✅ configs/
   ✅ docs/
   ```

4. **Repository Settings**
   - About 섹션에 Description 추가
   - Topics 추가: `kaggle`, `machine-learning`, `nlp`, `lora`, `debugging`

---

## 🎯 추천 방법

**처음 사용:** 방법 2 (GitHub Desktop) - 가장 쉬움
**빠른 업로드:** 방법 1 (Python API) - 자동화
**Git 익숙:** 방법 3 (Git CLI) - 전통적 방법

---

**마지막 업데이트:** 2024-10-25
