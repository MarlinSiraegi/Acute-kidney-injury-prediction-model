import pandas as pd
import numpy as np
from functools import reduce
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import os
import gc
from glob import glob

# 파일 경로 설정 (실제 경로로 수정하세요)
admissions_file = '/path/to/mimic-iv/hosp/admissions.csv.gz'
chartevents_file = '/path/to/mimic-iv/icu/chartevents.csv.gz'
labevents_file = '/path/to/mimic-iv/hosp/labevents.csv.gz'
diagnoses_file = '/path/to/mimic-iv/hosp/diagnoses_icd.csv.gz'

# 필요한 CSV 파일 로드
print("Loading admissions data...")
admissions = pd.read_csv(admissions_file)
print("Loading diagnoses data...")
diagnoses = pd.read_csv(diagnoses_file)

# AKI 진단을 위한 ICD-9 코드 리스트
aki_icd_codes = ['584', '5845', '5846', '5847', '5848', '5849']

# ICD 코드의 데이터 타입을 문자열로 변환
diagnoses['icd_code'] = diagnoses['icd_code'].astype(str)

# AKI 발병 환자 식별
diagnoses['is_aki'] = diagnoses['icd_code'].str.startswith(tuple(aki_icd_codes)).astype(int)
aki_patients = diagnoses[diagnoses['is_aki'] == 1]['subject_id'].unique()

# 입원 정보에 AKI 여부 라벨 추가
admissions['is_aki'] = admissions['subject_id'].isin(aki_patients).astype(int)

# 특징별 itemid 매핑
feature_itemids = {
    '혈청크레아티닌': [50912],
    '염소': [50902],
    '나트륨': [50983],
    '중탄산염': [50882],
    '포도당': [50931],
    '소변 비중': [51146],
    '칼륨': [50971],
    '혈소판 수치': [51265],
    '알칼리성 인산분해효소': [50863],
    '헤모글로빈': [51222],
    '알부민': [50862],
    'C-반응성 단백질 (CRP)': [51003],
    '프로트롬빈 시간': [51274],
    '국제 정규화 비율 (INR)': [51237],
    '활성화 부분 트롬보플라스틴 시간 (aPTT)': [51275],
    '마그네슘': [50960],
    '칼슘 (총 칼슘)': [50893],
    '인산': [50970],
    '이온화 칼슘': [50808]
}

# 필요한 모든 itemid를 합집합으로 모읍니다.
all_itemids = []
for ids in feature_itemids.values():
    all_itemids.extend(ids)
all_itemids = list(set(all_itemids))

# 필요한 컬럼 지정
chartevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
labevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']

# 청크 크기 설정
chunksize = 10 ** 6  # 1,000,000행씩 처리

# 입원 정보에서 필요한 컬럼만 선택하고 복사본 생성
admissions_subset = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'is_aki']].copy()
admissions_subset['admittime'] = pd.to_datetime(admissions_subset['admittime'])
admissions_subset['dischtime'] = pd.to_datetime(admissions_subset['dischtime'])

# 특징별로 데이터 유형 구분 (chartevents 또는 labevents)
feature_sources = {}
for feature_name in feature_itemids:
    if feature_name in ['혈청크레아티닌', '염소', '나트륨', '중탄산염', '포도당', '소변 비중', 
                        '칼륨', '혈소판 수치', '알칼리성 인산분해효소', '헤모글로빈', '알부민', 
                        'C-반응성 단백질 (CRP)', '프로트롬빈 시간', '국제 정규화 비율 (INR)', 
                        '활성화 부분 트롬보플라스틴 시간 (aPTT)', '마그네슘', '칼슘 (총 칼슘)', 
                        '인산', '이온화 칼슘']:
        feature_sources[feature_name] = 'labevents'
    else:
        feature_sources[feature_name] = 'chartevents'

# 임시 디렉토리 생성
temp_dir = 'temp_feature_aggregations'
os.makedirs(temp_dir, exist_ok=True)

# 특징별로 집계 결과를 저장할 리스트 초기화
merged_features = []

# labevents 처리
print("Processing labevents in chunks...")
labevents_iter = pd.read_csv(
    labevents_file,
    usecols=labevents_cols,
    chunksize=chunksize,
    low_memory=False
)

for i, chunk in enumerate(tqdm(labevents_iter)):
    # 필요한 itemid만 필터링
    chunk = chunk[chunk['itemid'].isin(all_itemids)]
    if chunk.empty:
        continue
    # 입원 정보와 병합
    chunk = chunk.merge(
        admissions_subset,
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    # datetime 형식으로 변환
    chunk['charttime'] = pd.to_datetime(chunk['charttime'])
    # 입원 후 첫 48시간 내의 데이터만 사용
    chunk = chunk[
        (chunk['charttime'] >= chunk['admittime']) & 
        (chunk['charttime'] <= chunk['dischtime'])
    ]
    # 결측치 제거
    chunk = chunk.dropna(subset=['valuenum'])
    # 이상치 제거
    chunk = chunk[chunk['valuenum'] >= 0]
    
    # 시간별 데이터 보존
    for feature_name, itemids in feature_itemids.items():
        if feature_sources[feature_name] == 'labevents':
            feature_chunk = chunk[chunk['itemid'].isin(itemids)]
            if feature_chunk.empty:
                continue
            
            # 시간별 데이터를 구성
            time_series_data = feature_chunk[['subject_id', 'hadm_id', 'valuenum']].copy()
            time_series_data = time_series_data.rename(columns={'valuenum': feature_name})
    
            # 임시 파일로 저장
            temp_file = os.path.join(temp_dir, f"{feature_name}_labevents_chunk_{i}.csv")
            time_series_data.to_csv(temp_file, index=False)
            
    # 청크 처리 후 메모리 해제
    del chunk
    gc.collect()

print("Merging aggregated features from temporary files...")
merged_features = []

for feature_name in feature_itemids.keys():
    # labevents 파일 읽기
    temp_files = glob(os.path.join(temp_dir, f"{feature_name}_labevents_chunk_*.csv"))
    
    if not temp_files:
        continue
    # 모든 임시 파일 읽기
    feature_dfs = [pd.read_csv(f) for f in temp_files]
    feature_df = pd.concat(feature_dfs, ignore_index=True)
    
    # 'subject_id', 'hadm_id' 기준으로 평균과 표준편차 집계
    feature_grouped = feature_df.groupby(['subject_id', 'hadm_id'])
    feature_mean = feature_grouped[feature_name].mean().reset_index()
    
    # 혈청크레아티닌인 경우 mean과 std 따로 처리
    if feature_name == '혈청크레아티닌':
        feature_std = feature_grouped[feature_name].std().reset_index()
        # 표준편차가 NaN인 경우(데이터가 하나인 경우)를 0으로 대체
        feature_std[feature_name] = feature_std[feature_name].fillna(0)
        feature_combined = feature_mean.copy()
        feature_combined[f'{feature_name}_mean'] = feature_mean[feature_name]
        feature_combined[f'{feature_name}_std'] = feature_std[feature_name]
        merged_features.append(feature_combined[['subject_id', 'hadm_id', f'{feature_name}_mean', f'{feature_name}_std']])
    else:
        merged_features.append(feature_mean[['subject_id', 'hadm_id', feature_name]])
    
    # 임시 파일 삭제
    for f in temp_files:
        os.remove(f)

# 모든 특징별 데이터를 병합
if merged_features:
    final_df = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'hadm_id'], how='outer'), merged_features)
else:
    final_df = pd.DataFrame()
    print("No features to merge.")

# AKI 라벨 추가
final_df = final_df.merge(
    admissions_subset[['subject_id', 'hadm_id', 'is_aki']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

# 결측치 처리
print("Handling missing values...")

# 결측치인 셀을 특정 값으로 대체 (예: '0')
feature_cols = final_df.columns.difference(['subject_id', 'hadm_id', 'is_aki'])
final_df[feature_cols] = final_df[feature_cols].fillna(0)

# 결과를 CSV 파일로 저장
output_file = 'data.csv'
final_df.to_csv(output_file, index=False)
print(f"Filtered features saved to {output_file}")

# 임시 디렉토리 삭제
try:
    os.rmdir(temp_dir)
    print(f"Temporary directory '{temp_dir}' deleted successfully.")
except OSError as e:
    print(f"Error deleting temporary directory '{temp_dir}': {e}")