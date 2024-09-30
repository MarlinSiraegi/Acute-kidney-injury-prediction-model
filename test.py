import pandas as pd
from pytorch_tabular import TabularModel
from sklearn.preprocessing import QuantileTransformer

# 모델이 저장된 폴더 경로와 테스트할 데이터 파일 경로
model_dir = "model"  # 모델이 저장된 폴더명
test_data_path = "data.csv"  # 테스트할 데이터 파일 경로

# 데이터 로드
data = pd.read_csv(test_data_path)

# 타겟과 특징 열 정의
target_name = 'is_aki'
feature_cols = data.columns.difference(['subject_id', 'hadm_id', target_name])

# 데이터 전처리
qt = QuantileTransformer(output_distribution='normal', random_state=42)
data[feature_cols] = qt.fit_transform(data[feature_cols])
data[target_name] = data[target_name].astype(int)

# TabularModel에서 모델 불러오기
tabular_model = TabularModel.load_model(model_dir)

# 모델 평가
result = tabular_model.evaluate(data)

# 평가 결과 출력
print("Final Model Performance:")
if isinstance(result, list):
    for metric, value in result[0].items():
        print(f"{metric}: {value:.4f}")
elif isinstance(result, dict):
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")
else:
    raise ValueError(f"Unexpected result format: {type(result)}")
