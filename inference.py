import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, CatBoostRegressor
import argparse
from train import models_directory, binary_target_columns, regression_target_columns, bin_to_reg_mapping, extract_features

def parse_arguments():
    parser = argparse.ArgumentParser(description="Генерация предсказаний для файлов ГДИС.")
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с входными файлами")
    parser.add_argument("--output_file", type=str, required=True, help="Имя выходного CSV-файла")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    input_directory = args.input_dir
    output_filename = args.output_file

    # Загрузка обученных моделей
    classifiers = {}
    for label_name in binary_target_columns:
        model_path = os.path.join(models_directory, f'classifier_{label_name}.cbm')
        clf = CatBoostClassifier()
        clf.load_model(model_path)
        classifiers[label_name] = clf

    regressors = {}
    for reg_target in regression_target_columns:
        model_path = os.path.join(models_directory, f'regressor_{reg_target}.cbm')
        reg = CatBoostRegressor()
        reg.load_model(model_path)
        regressors[reg_target] = reg

    # Подготовка данных для инференса
    validation_files = os.listdir(input_directory)
    features_list, valid_files = [], []
    for fname in tqdm(validation_files, desc="Извлечение признаков"):
        features = extract_features(fname, data_folder=input_directory)
        
        if features is not None:
            features_list.append(features)
            valid_files.append(fname)

    X_valid = pd.DataFrame(features_list)
    pred_bin = np.zeros((len(X_valid), len(classifiers)), dtype=int)
    pred_reg = np.full((len(X_valid), len(regressors)), np.nan)

    X_valid_bin = X_valid.copy()
    thresholds = pd.read_csv('thr.csv').to_dict()

    # Генерация бинарных предсказаний
    for idx, (target, clf) in enumerate(classifiers.items()):
        probabilities = clf.predict_proba(X_valid_bin)[:, 1]
        thr = thresholds.get(label_name, 0.5) # Порог 0.5 по умолчанию
        pred_bin[:, idx] = (probabilities > thr).astype(int)
        X_valid_bin[target] = pred_bin[:, idx]

    # Генерация регрессионных предсказаний
    for reg_target, reg in regressors.items():
        reg_idx = list(regressors.keys()).index(reg_target)
        bin_idx = [k for k, v in bin_to_reg_mapping.items() if v == reg_idx][0]
        mask = (pred_bin[:, bin_idx] == 1)
        X_subset = X_valid[mask]
        if len(X_subset) > 0:
            pred_reg[mask, reg_idx] = reg.predict(X_subset)

    # Формирование итогового CSV
    submission_columns = ["file_name"] + list(classifiers.keys()) + list(regressors.keys())
    submission_data = []
    for fname in validation_files:
        row = {"file_name": fname}
        if fname not in valid_files or X_valid.loc[valid_files.index(fname), "n_points"] < 15:
            # Значения по умолчанию
            for target in classifiers:
                row[target] = 1 if target == "Некачественное ГДИС" else 0
            for reg_target in regressors:
                row[reg_target] = 0.0
        else:
            idx = valid_files.index(fname)
            for target_idx, target in enumerate(classifiers):
                row[target] = int(pred_bin[idx, target_idx])
            for reg_idx, reg_target in enumerate(regressors):
                row[reg_target] = float(pred_reg[idx, reg_idx]) if not np.isnan(pred_reg[idx, reg_idx]) else 0.0
        submission_data.append(row)

    submission_df = pd.DataFrame(submission_data, columns=submission_columns)
    submission_df.to_csv(output_filename, index=False)
    print(f"Результат сохранен в {output_filename}")