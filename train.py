import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# PyTorch 및 PyTorch Tabular 라이브러리 임포트
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig

# 데이터 로드
data = pd.read_csv('/path/to/data.csv')

# 특징과 타겟 분리
target_name = 'is_aki'
feature_cols = data.columns.difference(['subject_id', 'hadm_id', target_name])

# 데이터 분할: Train (80%), Temp (20%)
train_df, temp_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data[target_name]
)

# Temp 데이터를 Validation (10%)과 Test (10%)로 분할
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df[target_name]
)

# 연속형 변수 변환 적용 (QuantileTransformer)
qt = QuantileTransformer(output_distribution='normal', random_state=42)
train_df[feature_cols] = qt.fit_transform(train_df[feature_cols])
valid_df[feature_cols] = qt.transform(valid_df[feature_cols])
test_df[feature_cols] = qt.transform(test_df[feature_cols])

# 타겟값을 정수형으로 변환
train_df[target_name] = train_df[target_name].astype(int)
valid_df[target_name] = valid_df[target_name].astype(int)
test_df[target_name] = test_df[target_name].astype(int)

# 타겟값 확인
print("Train target unique values:", train_df[target_name].unique())
print("Validation target unique values:", valid_df[target_name].unique())
print("Test target unique values:", test_df[target_name].unique())

# 최적의 하이퍼파라미터
best_params = {
    'learning_rate': 0.0005,
    'input_embed_dim': 64,       # 임베딩 차원 증가
    'num_attn_blocks': 8,         # 어텐션 블록 수 증가
    'num_heads': 8,               # 헤드 수 증가
    'attn_dropout': 0.1,          # 드롭아웃
    'ff_dropout': 0.1,            # 드롭아웃
    'batch_size': 512
}

# ExperimentConfig 설정
experiment_config = ExperimentConfig(
    project_name="FTTransformer_Project",  # 프로젝트 이름
    run_name="FTTransformer_Run",           # 실행 이름
    log_target="tensorboard",               # 로깅 타겟
)

# TrainerConfig 설정
trainer_config = TrainerConfig(
    max_epochs=100,  # 에포크 수 조정
    min_epochs=1,
    early_stopping='valid_loss',
    early_stopping_patience=25,
    early_stopping_min_delta=0.001,
    early_stopping_mode='min',
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None,
    batch_size=best_params['batch_size'],
    trainer_kwargs={},
)

# 데이터 구성 설정
data_config = DataConfig(
    target=[target_name],
    continuous_cols=feature_cols.tolist(),
    categorical_cols=[],
    normalize_continuous_features=True,
    num_workers=14,
)

# Optimizer 설정
optimizer_config = OptimizerConfig(
    optimizer="AdamW",
    optimizer_params={"weight_decay": 1e-5},
    lr_scheduler="ReduceLROnPlateau",
    lr_scheduler_params={
        "mode": "min",
        "factor": 0.05,
        "patience": 5,
        "threshold": 0.0001,
    },
    lr_scheduler_monitor_metric="valid_loss",
)

# 모델 구성 설정
model_config = FTTransformerConfig(
    task="classification",
    learning_rate=best_params['learning_rate'],
    loss="CrossEntropyLoss",
    metrics=["accuracy", "f1_score", "auroc"],
    metrics_params=[{}, {}, {}],
    metrics_prob_input=[False, False, True],
    input_embed_dim=best_params['input_embed_dim'],
    embedding_dims=[],
    num_attn_blocks=best_params['num_attn_blocks'],
    num_heads=best_params['num_heads'],
    attn_dropout=best_params['attn_dropout'],
    ff_dropout=best_params['ff_dropout'],
    head="LinearHead",
    head_config={},
)

# 모델 초기화
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
    verbose=False,
    suppress_lightning_logger=True,
)

# 불필요한 컬럼 제거
train_data = train_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore')
valid_data = valid_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore')

# 모델 학습
tabular_model.fit(train=train_data, validation=valid_data)

# 모델 저장
tabular_model.save_model("model")
print("Model successfully saved in 'model'")