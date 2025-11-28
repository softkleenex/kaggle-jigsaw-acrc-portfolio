"""
Jigsaw ACRC BERT Fine-tuning Model
===================================
This script implements a BERT-based fine-tuning model for the Jigsaw ACRC competition using:
- transformers library (Hugging Face)
- bert-base-uncased or roberta-base
- Input format: [CLS] body [SEP] rule [SEP] (with optional positive/negative examples)
- Stratified 5-Fold Cross-Validation
- ROC-AUC scoring
- GPU support (automatic fallback to CPU if unavailable)

Author: Claude Code
Date: 2025-10-13
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# 랜덤 시드 설정 (재현성을 위해)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# 모델 설정
MODEL_NAME = 'bert-base-uncased'  # 또는 'roberta-base'
# MODEL_NAME = 'roberta-base'  # RoBERTa를 사용하려면 이 줄의 주석을 해제하고 위 줄을 주석 처리

# 학습 하이퍼파라미터
BATCH_SIZE = 16  # GPU 메모리에 따라 조정 (8, 16, 32 등)
MAX_LENGTH = 256  # 토큰 최대 길이 (메모리와 속도 고려)
EPOCHS = 3  # 에폭 수 (BERT는 보통 2-4 에폭이면 충분)
LEARNING_RATE = 2e-5  # BERT fine-tuning 권장 학습률 (2e-5, 3e-5, 5e-5)
WARMUP_RATIO = 0.1  # 웜업 스텝 비율
WEIGHT_DECAY = 0.01  # L2 정규화
MAX_GRAD_NORM = 1.0  # Gradient clipping

# Cross-Validation 설정
N_FOLDS = 5  # Stratified K-Fold 수

# Examples 포함 여부 (True: 포함, False: body+rule만)
# 주의: Examples를 포함하면 토큰 길이가 길어져 MAX_LENGTH 초과 가능
INCLUDE_EXAMPLES = False  # 기본값은 False (메모리 절약)

# GPU 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 경로
DATA_DIR = '/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# 출력 경로
OUTPUT_DIR = '/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/submissions'
MODEL_SAVE_DIR = '/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/models/bert'

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """
    학습, 테스트, 제출 샘플 데이터를 로드합니다.

    Returns:
        tuple: (train_df, test_df, sample_submission)
    """
    print("="*80)
    print("Loading data...")
    print("="*80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    print(f"\nTarget distribution:")
    print(train_df['rule_violation'].value_counts())
    print(f"Positive rate: {train_df['rule_violation'].mean():.4f}")

    return train_df, test_df, sample_submission


def create_input_text(row, include_examples=False):
    """
    데이터프레임의 한 행을 BERT 입력 형식으로 변환합니다.

    입력 형식:
    - 기본: [CLS] body [SEP] rule [SEP]
    - Examples 포함: [CLS] body [SEP] rule [SEP] positive_examples [SEP] negative_examples [SEP]

    Args:
        row: 데이터프레임의 한 행
        include_examples: positive/negative examples 포함 여부

    Returns:
        str: 조합된 입력 텍스트
    """
    # 기본: body + rule
    body = str(row['body']) if pd.notna(row['body']) else ""
    rule = str(row['rule']) if pd.notna(row['rule']) else ""

    # 기본 형식: body [SEP] rule
    text = f"{body} [SEP] {rule}"

    # Examples 포함 (선택적)
    if include_examples:
        # Positive examples
        pos_examples = []
        for col in ['positive_example_1', 'positive_example_2']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                pos_examples.append(str(row[col]).strip())

        # Negative examples
        neg_examples = []
        for col in ['negative_example_1', 'negative_example_2']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                neg_examples.append(str(row[col]).strip())

        # Positive examples 추가
        if pos_examples:
            pos_text = " ".join(pos_examples)
            text += f" [SEP] {pos_text}"

        # Negative examples 추가
        if neg_examples:
            neg_text = " ".join(neg_examples)
            text += f" [SEP] {neg_text}"

    return text


def prepare_texts(df, include_examples=False):
    """
    데이터프레임 전체를 입력 텍스트로 변환합니다.

    Args:
        df: 데이터프레임
        include_examples: examples 포함 여부

    Returns:
        list: 입력 텍스트 리스트
    """
    texts = []
    for idx, row in df.iterrows():
        text = create_input_text(row, include_examples=include_examples)
        texts.append(text)

    return texts


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class JigsawDataset(Dataset):
    """
    Jigsaw ACRC 데이터셋을 위한 PyTorch Dataset 클래스

    이 클래스는 텍스트를 토큰화하고 BERT 입력 형식으로 변환합니다.
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Args:
            texts: 입력 텍스트 리스트
            labels: 레이블 리스트 (없으면 None)
            tokenizer: Hugging Face tokenizer
            max_length: 최대 토큰 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 토큰화
        # - padding: 최대 길이까지 패딩
        # - truncation: 최대 길이 초과 시 자르기
        # - return_tensors: PyTorch tensor로 반환
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS], [SEP] 자동 추가
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        # 레이블이 있으면 추가
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    한 에폭 학습을 수행합니다.

    Args:
        model: BERT 모델
        dataloader: 학습 데이터로더
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        device: 디바이스 (cuda/cpu)

    Returns:
        float: 평균 손실값
    """
    model.train()
    total_loss = 0

    # tqdm으로 진행상황 표시
    progress_bar = tqdm(dataloader, desc='Training')

    for batch in progress_bar:
        # 배치 데이터를 디바이스로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Gradient 초기화
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping (폭발 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # 가중치 업데이트
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 진행상황 업데이트
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """
    검증 데이터에 대한 평가를 수행합니다.

    Args:
        model: BERT 모델
        dataloader: 검증 데이터로더
        device: 디바이스 (cuda/cpu)

    Returns:
        tuple: (예측 확률 리스트, 실제 레이블 리스트, 평균 손실)
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0

    # tqdm으로 진행상황 표시
    progress_bar = tqdm(dataloader, desc='Evaluating')

    # Gradient 계산 비활성화 (메모리 절약, 속도 향상)
    with torch.no_grad():
        for batch in progress_bar:
            # 배치 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 손실 계산
            loss = outputs.loss
            total_loss += loss.item()

            # 예측 확률 계산 (sigmoid 적용)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.extend(probs.flatten().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)

    return predictions, true_labels, avg_loss


def predict(model, dataloader, device):
    """
    테스트 데이터에 대한 예측을 수행합니다.

    Args:
        model: BERT 모델
        dataloader: 테스트 데이터로더
        device: 디바이스 (cuda/cpu)

    Returns:
        list: 예측 확률 리스트
    """
    model.eval()
    predictions = []

    # tqdm으로 진행상황 표시
    progress_bar = tqdm(dataloader, desc='Predicting')

    # Gradient 계산 비활성화
    with torch.no_grad():
        for batch in progress_bar:
            # 배치 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 예측 확률 계산 (sigmoid 적용)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.extend(probs.flatten().tolist())

    return predictions


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def train_with_cross_validation(train_df, tokenizer, n_splits=5):
    """
    Stratified K-Fold Cross-Validation으로 모델을 학습하고 평가합니다.

    Args:
        train_df: 학습 데이터프레임
        tokenizer: Hugging Face tokenizer
        n_splits: Fold 수

    Returns:
        tuple: (CV 점수 리스트, Out-of-Fold 예측값, Fold별 모델 리스트)
    """
    print("\n" + "="*80)
    print(f"Training with {n_splits}-Fold Stratified Cross-Validation")
    print("="*80)

    # 입력 텍스트 준비
    print(f"\nPreparing texts (include_examples={INCLUDE_EXAMPLES})...")
    texts = prepare_texts(train_df, include_examples=INCLUDE_EXAMPLES)
    labels = train_df['rule_violation'].values

    # Stratified K-Fold 초기화
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # 결과 저장
    cv_scores = []
    oof_predictions = np.zeros(len(train_df))
    fold_models = []

    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print("\n" + "-"*80)
        print(f"Fold {fold}/{n_splits}")
        print("-"*80)

        # Train/Validation split
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
        print(f"Train positive rate: {train_labels.mean():.4f}")
        print(f"Validation positive rate: {val_labels.mean():.4f}")

        # Dataset & DataLoader 생성
        train_dataset = JigsawDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = JigsawDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Windows에서 문제 발생 시 0으로 설정
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # 모델 초기화
        print(f"\nInitializing model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,  # Binary classification (sigmoid 사용)
            problem_type="regression"  # BCEWithLogitsLoss 사용
        )
        model.to(DEVICE)

        # Optimizer & Scheduler 설정
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Total training steps
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        # 에폭별 학습
        best_auc = 0
        best_predictions = None

        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

            # 학습
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            print(f"Train Loss: {train_loss:.6f}")

            # 검증
            val_predictions, val_true_labels, val_loss = evaluate(model, val_loader, DEVICE)
            val_auc = roc_auc_score(val_true_labels, val_predictions)

            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Validation AUC: {val_auc:.6f}")

            # Best 모델 저장
            if val_auc > best_auc:
                best_auc = val_auc
                best_predictions = val_predictions.copy()
                print(f">>> Best AUC improved: {best_auc:.6f}")

        # Fold 결과 저장
        cv_scores.append(best_auc)
        oof_predictions[val_idx] = best_predictions
        fold_models.append(model)

        print(f"\nFold {fold} Best AUC: {best_auc:.6f}")

    # Overall CV 점수 계산
    overall_auc = roc_auc_score(labels, oof_predictions)

    # 결과 출력
    print("\n" + "="*80)
    print("Cross-Validation Results")
    print("="*80)
    for fold, score in enumerate(cv_scores, 1):
        print(f"Fold {fold} AUC: {score:.6f}")
    print(f"\nMean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(f"Overall CV AUC (OOF): {overall_auc:.6f}")
    print("="*80)

    return cv_scores, oof_predictions, fold_models


# =============================================================================
# TEST PREDICTION
# =============================================================================

def predict_test(test_df, tokenizer, fold_models):
    """
    테스트 데이터에 대한 예측을 수행합니다.
    여러 Fold 모델의 예측을 평균냅니다 (앙상블).

    Args:
        test_df: 테스트 데이터프레임
        tokenizer: Hugging Face tokenizer
        fold_models: Fold별 학습된 모델 리스트

    Returns:
        np.array: 최종 예측 확률
    """
    print("\n" + "="*80)
    print("Predicting on test data")
    print("="*80)

    # 입력 텍스트 준비
    print(f"Preparing test texts (include_examples={INCLUDE_EXAMPLES})...")
    test_texts = prepare_texts(test_df, include_examples=INCLUDE_EXAMPLES)

    # Dataset & DataLoader 생성
    test_dataset = JigsawDataset(test_texts, None, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 각 Fold 모델로 예측 후 평균
    all_predictions = []

    for fold, model in enumerate(fold_models, 1):
        print(f"\nPredicting with Fold {fold} model...")
        predictions = predict(model, test_loader, DEVICE)
        all_predictions.append(predictions)

    # 앙상블: 평균
    final_predictions = np.mean(all_predictions, axis=0)

    print(f"\nTest prediction completed!")
    print(f"Prediction shape: {final_predictions.shape}")
    print(f"Prediction range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")

    return final_predictions


# =============================================================================
# SUBMISSION
# =============================================================================

def create_submission(test_df, predictions, sample_submission, timestamp):
    """
    제출 파일을 생성합니다.

    Args:
        test_df: 테스트 데이터프레임
        predictions: 예측 확률
        sample_submission: 샘플 제출 파일
        timestamp: 타임스탬프

    Returns:
        str: 저장된 제출 파일 경로
    """
    print("\n" + "="*80)
    print("Creating submission file")
    print("="*80)

    # 제출 파일 생성
    submission = sample_submission.copy()
    submission['rule_violation'] = predictions

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 파일명 생성
    model_short_name = MODEL_NAME.split('/')[-1]
    examples_suffix = "_with_examples" if INCLUDE_EXAMPLES else ""
    filename = f"bert_{model_short_name}{examples_suffix}_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # 저장
    submission.to_csv(output_path, index=False)

    print(f"Submission file saved: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10))

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    메인 실행 함수
    """
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "="*80)
    print("JIGSAW ACRC - BERT FINE-TUNING MODEL")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Include examples: {INCLUDE_EXAMPLES}")
    print(f"N-Folds: {N_FOLDS}")

    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU not available, using CPU")

    # 1. 데이터 로드
    train_df, test_df, sample_submission = load_data()

    # 2. Tokenizer 초기화
    print("\n" + "="*80)
    print("Initializing tokenizer")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")

    # 샘플 텍스트 토큰화 예시
    sample_text = create_input_text(train_df.iloc[0], include_examples=INCLUDE_EXAMPLES)
    sample_tokens = tokenizer.encode(sample_text, truncation=True, max_length=MAX_LENGTH)
    print(f"\nSample text length: {len(sample_text)} chars")
    print(f"Sample tokens length: {len(sample_tokens)} tokens")
    print(f"Sample text (first 200 chars): {sample_text[:200]}...")

    # 3. Cross-Validation 학습
    cv_scores, oof_predictions, fold_models = train_with_cross_validation(
        train_df,
        tokenizer,
        n_splits=N_FOLDS
    )

    # 4. 테스트 데이터 예측
    test_predictions = predict_test(test_df, tokenizer, fold_models)

    # 5. 제출 파일 생성
    submission_path = create_submission(test_df, test_predictions, sample_submission, timestamp)

    # 6. 모델 저장 (선택적)
    # 마지막 Fold 모델 저장 (또는 모든 Fold 모델 저장 가능)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    model_save_path = os.path.join(MODEL_SAVE_DIR, f"bert_model_{timestamp}")
    fold_models[-1].save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nLast fold model saved: {model_save_path}")

    # 실행 시간 계산
    elapsed_time = time.time() - start_time

    # 최종 요약
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Input format: [CLS] body [SEP] rule [SEP]" +
          (" [SEP] examples [SEP]" if INCLUDE_EXAMPLES else ""))
    print(f"\nHyperparameters:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Warmup ratio: {WARMUP_RATIO}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - Max grad norm: {MAX_GRAD_NORM}")
    print(f"\nCross-Validation Results:")
    print(f"  - Method: Stratified {N_FOLDS}-Fold")
    for fold, score in enumerate(cv_scores, 1):
        print(f"  - Fold {fold} AUC: {score:.6f}")
    print(f"  - Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    overall_auc = roc_auc_score(train_df['rule_violation'].values, oof_predictions)
    print(f"  - Overall CV AUC (OOF): {overall_auc:.6f}")
    print(f"\nExecution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Submission file: {submission_path}")
    print(f"Model saved: {model_save_path}")
    print("="*80)

    return {
        'cv_scores': cv_scores,
        'mean_cv_auc': np.mean(cv_scores),
        'std_cv_auc': np.std(cv_scores),
        'overall_cv_auc': overall_auc,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'submission_path': submission_path,
        'model_save_path': model_save_path,
        'elapsed_time': elapsed_time
    }


if __name__ == '__main__':
    # 실행
    results = main()

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check the submission file and submit to Kaggle")
    print("2. Try different hyperparameters (learning rate, epochs, max_length)")
    print("3. Try including examples (set INCLUDE_EXAMPLES=True)")
    print("4. Try different models (roberta-base, bert-large, etc.)")
    print("5. Ensemble with other models (baseline, SetFit, etc.)")
    print("="*80)
