import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
general_markup = pd.read_csv('markup_train.csv')
data_directory = 'data/'

# Создаем директорию для сохранения моделей
models_directory = 'saved_models/'
os.makedirs(models_directory, exist_ok=True)

# Бинарные метки для классификации
binary_target_columns = [
    'Некачественное ГДИС',
    'Влияние ствола скважины',
    'Радиальный режим',
    'Линейный режим',
    'Билинейный режим',
    'Сферический режим',
    'Граница постоянного давления',
    'Граница непроницаемый разлом'
]

# Вещественные метки для регрессии
regression_target_columns = [
    'Влияние ствола скважины_details',
    'Радиальный режим_details',
    'Линейный режим_details',
    'Билинейный режим_details',
    'Сферический режим_details',
    'Граница постоянного давления_details',
    'Граница непроницаемый разлом_details'
]

# Mapping для регрессии
bin_to_reg_mapping = {
    1: 0,  # Влияние ствола скважины
    2: 1,  # Радиальный режим
    3: 2,  # Линейный режим
    4: 3,  # Билинейный режим
    5: 4,  # Сферический режим
    6: 5,  # Граница постоянного давления
    7: 6   # Граница непроницаемый разлом
}

def preprocess(dataframe):
    # Удаление строк, где только "Некачественное ГДИС" = 1, а остальные метки = 0
    filter_mask = (
        (dataframe['Некачественное ГДИС'] == 1) &
        (dataframe[binary_target_columns[1:]].sum(axis=1) == 0)
    )
    filtered_dataframe = dataframe[~filter_mask].reset_index(drop=True)
    print(f"Оставшиеся строки после фильтрации: {len(filtered_dataframe)}")
    return filtered_dataframe

def find_best_threshold(y_true, y_probs):
    """Находит лучший порог по Precision Recall кривой на основе максимизации F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)  # Избегаем деления на ноль
    best_index = np.argmax(f1_scores)
    return thresholds[best_index], f1_scores[best_index]

def adjust_predictions(y_prob, threshold, X_test, y_reg_test, reg=None, reg_idx=None):
    """Функция корректировки предсказаний с учётом условий задачи"""
    pred = (y_prob > threshold).astype(int)
    if reg is not None:
        pos_idx = np.where(pred == 1)[0]
        if len(pos_idx) > 0:
            # Если X_test – DataFrame, используем .iloc для выборки строк
            X_test_rows = X_test.iloc[pos_idx] if hasattr(X_test, 'iloc') else X_test[pos_idx]
            reg_pred = reg.predict(X_test_rows)
            # Если y_reg_test – DataFrame, аналогично используем .iloc
            if hasattr(y_reg_test, 'iloc'):
                true_reg = y_reg_test.iloc[pos_idx, reg_idx]
            else:
                true_reg = y_reg_test[pos_idx, reg_idx]
            errors = np.abs(true_reg - reg_pred)
            # Для тех примеров, где ошибка > 0.15, сбрасываем предсказание в 0
            pred[pos_idx] = (errors <= 0.15).astype(int)
    return pred

def extract_features(file_name, data_folder="data", segments=5):
    """Извлекает признаки из файла с данными ГДИС."""
    file_path = os.path.join(data_folder, file_name)
    if not os.path.exists(file_path):
        return None
    try:
        data = np.loadtxt(file_path)
        time = data[:, 0]
        pressure_diff = data[:, 1]
        derivative = data[:, 2]
    except:
        return None

    valid_mask = (time > 0)
    time = time[valid_mask]
    pressure_diff = pressure_diff[valid_mask]
    derivative = derivative[valid_mask]
    if len(time) < 5:
        return None

    features = {
        'n_points': len(time),
        'time_min': float(time.min()),
        'time_max': float(time.max()),
        'p_diff_mean': float(pressure_diff.mean()),
        'p_diff_std': float(pressure_diff.std()),
        'p_der_mean': float(derivative.mean()),
        'p_der_std': float(derivative.std()),
    }

    # Коэффициент вариации
    if abs(derivative.mean()) > 1e-9:
        features['p_der_cv'] = float(derivative.std() / abs(derivative.mean()))
    else:
        features['p_der_cv'] = 0.0 # TO-DO: Почему ноль? Если среднее маленькое => CV -> infinity

    log_time = np.log10(time)
    if np.isclose(log_time.min(), log_time.max()):
        return None

    bounds = np.linspace(log_time.min(), log_time.max(), segments + 1) # TO-DO: Число сегментов равно число 0.5 лог-циклов
    for i in range(segments):
        segment_mask = (log_time >= bounds[i]) & (log_time < bounds[i+1])
        seg_lt = log_time[segment_mask]

        if len(seg_lt) < 3:
            features.update({
                f"seg_{i}_dp_slope": 0.0,
                f"seg_{i}_dp_intercept": 0.0,
                f"seg_{i}_dp_mean": 0.0,
                f"seg_{i}_dr_slope": 0.0,
                f"seg_{i}_dr_intercept": 0.0
            })
            continue

        y_dp = np.log10(np.abs(pressure_diff[segment_mask]) + 1e-12)
        dp_slope, dp_intercept = np.polyfit(seg_lt, y_dp, 1)
        y_dr = np.log10(np.abs(derivative[segment_mask]) + 1e-12)
        dr_slope, dr_intercept = np.polyfit(seg_lt, y_dr, 1)

        features.update({
            f"seg_{i}_dp_mean": float(pressure_diff[segment_mask].mean()),
            f"seg_{i}_dp_slope": float(dp_slope),
            f"seg_{i}_dp_intercept": float(dp_intercept),
            f"seg_{i}_dr_slope": float(dr_slope),
            f"seg_{i}_dr_intercept": float(dr_intercept)
        })

    return features


if __name__ == '__main__':
    general_markup = preprocess(general_markup)

    features_list, binary_labels_list, regression_values_list, filenames = [], [], [], []
    for _, row in tqdm(general_markup.iterrows(), total=len(general_markup)):
        file_name = row['file_name']
        features = extract_features(file_name=file_name, data_folder=data_directory, segments=5)

        if features is None:
            continue

        features_list.append(features)
        binary_labels_list.append(row[binary_target_columns].values.astype(int))
        regression_values_list.append(row[regression_target_columns].values.astype(float))
        filenames.append(file_name)

    X = pd.DataFrame(features_list)
    Y_bin = np.array(binary_labels_list) # Shape: (N, 8)
    Y_reg = np.array(regression_values_list) # # Shape: (N, 7)

    X_train, X_test, y_bin_train, y_bin_test, y_reg_train, y_reg_test = train_test_split(
        X, Y_bin, Y_reg, test_size=0.2, random_state=42
    )

    # Обучение классификаторов
    classifiers = {}
    y_test_dict = {}
    for idx, target in enumerate(binary_target_columns):
        y_train = y_bin_train[:, idx]
        y_test_dict[target] = y_bin_test[:, idx] # Сохраняем истинные значения для теста
        class_weights = [1.0, (y_train == 0).sum() / (y_train == 1).sum()] if (y_train == 1).any() else None

        if class_weights is None:
            print('Все значения y_train равны нулю! Пропуск объекта.')
            continue

        clf = CatBoostClassifier(
            depth=6,
            loss_function='Logloss',
            verbose=False,
            class_weights=class_weights
        )
        clf.fit(X_train, y_train)
        classifiers[target] = clf

    # Обучение регрессоров
    regressors = {}
    for bin_idx, reg_idx in bin_to_reg_mapping.items():
        target_name = binary_target_columns[bin_idx]
        reg_target = regression_target_columns[reg_idx]
        train_mask = (y_bin_train[:, bin_idx] == 1)
        X_reg_train = X_train[train_mask]
        y_reg_train_subset = y_reg_train[train_mask, reg_idx]
        
        if len(X_reg_train) < 5:
            continue

        reg = CatBoostRegressor(
            depth=6,
            loss_function='RMSE',
            verbose=False
        )
        reg.fit(X_reg_train, y_reg_train_subset)
        regressors[reg_target] = reg

    # Оценка регрессоров на тестовой выборке
    test_mask = (y_bin_test[:, bin_idx] == 1)
    X_reg_test = X_test[test_mask]
    y_reg_test_subset = y_reg_test[test_mask, reg_idx]
    if len(X_reg_test) > 0:
        yhat_reg = reg.predict(X_reg_test)
        mae_test = mean_absolute_error(y_reg_test_subset, yhat_reg)
        r2_test  = r2_score(y_reg_test_subset, yhat_reg)
        print(f"Значения метрик на тесте для регресии {reg_target}, MAE={mae_test:.3f}, R2={r2_test:.3f}, Count={len(X_reg_test)}")
    
    # Для каждого классификатора подбираем порог, максимизирующий F1 с учётом условия по MAE.
    thresholds = {}
    f1_scores_all = {}

    for label_name, clf in classifiers.items():
        y_ts = y_test_dict[label_name]
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        reg = None
        reg_idx = bin_to_reg_mapping.get(binary_target_columns.index(label_name), None)
        if reg_idx is not None:
            reg_label = regression_target_columns[reg_idx]
            reg = regressors.get(reg_label)
        
        best_thr, best_f1 = find_best_threshold(y_ts, y_prob_test)
        thresholds[label_name] = best_thr
        f1_scores_all[label_name] = best_f1
        
        print(f"\n=== Признак '{label_name}' ===")
        print(f"Оптимальный порог: {best_thr:.2f}, F1: {best_f1:.3f}")
        final_pred = adjust_predictions(y_prob_test, best_thr, X_test, y_reg_test, reg, reg_idx)
        print(classification_report(y_ts, final_pred, zero_division=0))
        print("-" * 50)

    if f1_scores_all:
        mean_f1 = np.mean(list(f1_scores_all.values()))
        print(f"\nСредний F1 по всем классам: {mean_f1:.3f}")

    # Сохраняем классификаторы
    for label_name, clf in classifiers.items():
        model_path = os.path.join(models_directory, f'classifier_{label_name}.cbm')
        clf.save_model(model_path)
        print(f"Сохранен классификатор для '{label_name}' в {model_path}")

    # Сохраняем регрессионые модели
    for reg_target, reg in regressors.items():
        model_path = os.path.join(models_directory, f'regressor_{reg_target}.cbm')
        reg.save_model(model_path)
        print(f"Сохранен регрессор для '{reg_target}' в {model_path}")