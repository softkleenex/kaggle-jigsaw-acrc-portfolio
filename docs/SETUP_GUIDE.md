# Kaggle API 설정 가이드

## 1. Kaggle API 토큰 생성

1. https://www.kaggle.com 접속 및 로그인
2. 우측 상단 프로필 아이콘 클릭 → **Settings** 이동
3. **API** 섹션에서 **"Create New API Token"** 클릭
4. `kaggle.json` 파일이 자동으로 다운로드됩니다

## 2. kaggle.json 파일 배치

### Linux/WSL (현재 환경)

```bash
# .kaggle 디렉토리 생성 (두 위치 모두 생성)
mkdir -p ~/.kaggle
mkdir -p ~/.config/kaggle

# kaggle.json 파일을 다운로드 폴더에서 복사
# Windows 다운로드 폴더에서 복사 (경로 확인 필요)
cp /mnt/c/Users/YOUR_USERNAME/Downloads/kaggle.json ~/.config/kaggle/

# 또는 직접 경로 지정
# cp /mnt/c/LSJ/Downloads/kaggle.json ~/.config/kaggle/

# 파일 권한 설정 (필수!)
chmod 600 ~/.config/kaggle/kaggle.json

# 또는 ~/.kaggle/에도 복사 (이중 백업)
cp ~/.config/kaggle/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**참고**: 현재 시스템에서는 `~/.config/kaggle/kaggle.json` 경로를 사용합니다.

### Windows (참고용)

`kaggle.json` 파일을 다음 위치에 배치:
```
C:\Users\<사용자명>\.kaggle\kaggle.json
```

## 3. 인증 확인

```bash
kaggle competitions list
```

대회 목록이 출력되면 설정 완료!

## 4. 대회 데이터 다운로드

```bash
# 프로젝트 루트 디렉토리에서 실행
cd /mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC

# 대회 데이터 다운로드
kaggle competitions download -c jigsaw-agile-community-rules

# data 폴더에 압축 해제
unzip jigsaw-agile-community-rules.zip -d data/

# 압축 파일 삭제 (선택)
rm jigsaw-agile-community-rules.zip
```

## 5. Python 패키지 설치

```bash
pip install -r requirements.txt --user
```

## 6. 대회 참가 (중요!)

데이터를 다운로드하기 전에 Kaggle 웹사이트에서 대회에 참가해야 합니다:

1. https://www.kaggle.com/competitions/jigsaw-agile-community-rules 접속
2. **"Join Competition"** 버튼 클릭
3. 대회 규칙에 동의
4. 참가 완료 후 데이터 다운로드 가능

## 7. 제출 방법

### CLI로 제출
```bash
kaggle competitions submit -c jigsaw-agile-community-rules \
  -f submissions/submission.csv \
  -m "First submission"
```

### Python으로 제출
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.competition_submit(
    file_name='submissions/submission.csv',
    message='First submission',
    competition='jigsaw-agile-community-rules'
)
```

## 8. 제출 기록 확인

```bash
kaggle competitions submissions -c jigsaw-agile-community-rules
```

## Troubleshooting

### 문제: "Could not find kaggle.json"
- kaggle.json 파일이 `~/.kaggle/` 디렉토리에 있는지 확인
- 파일 권한이 600으로 설정되어 있는지 확인 (`chmod 600 ~/.kaggle/kaggle.json`)

### 문제: "403 - Forbidden"
- Kaggle 웹사이트에서 대회에 참가했는지 확인
- API 토큰이 유효한지 확인 (재발급 필요할 수 있음)

### 문제: "kaggle: command not found"
- PATH에 `~/.local/bin`이 추가되어 있는지 확인
- 추가 방법: `export PATH=$PATH:~/.local/bin`을 `~/.bashrc`에 추가

---

**다음 단계**: 설정이 완료되면 `notebooks/01_EDA.ipynb`에서 데이터 탐색을 시작하세요!
