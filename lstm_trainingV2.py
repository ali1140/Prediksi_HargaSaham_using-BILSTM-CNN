import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import glob
import matplotlib.pyplot as plt

def convert_volume_to_numeric(volume_str):
    if isinstance(volume_str, (int, float)):
        return volume_str
    if pd.isna(volume_str):
        return np.nan
    volume_str = str(volume_str).strip().upper()
    if not volume_str:
        return np.nan
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    for suffix, multiplier in multipliers.items():
        if volume_str.endswith(suffix):
            try:
                return float(volume_str[:-len(suffix)]) * multiplier
            except ValueError:
                return np.nan
    try:
        return float(volume_str)
    except ValueError:
        return np.nan

def convert_change_pct_to_numeric(change_str):
    if isinstance(change_str, (int, float)):
        return change_str
    if pd.isna(change_str):
        return np.nan
    change_str = str(change_str).strip()
    if not change_str:
        return np.nan
    try:
        return float(change_str.rstrip('%')) / 100.0
    except ValueError:
        return np.nan

def create_sequences_multi_feature(data, sequence_length, future_steps, target_column_index):
    X, y = [], []
    for i in range(len(data) - sequence_length - future_steps + 1):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[(i + sequence_length):(i + sequence_length + future_steps), target_column_index])
    if not X or not y:
        raise ValueError(
            "Tidak cukup data untuk membuat sekuens dengan sequence_length "
            f"({sequence_length}) dan future_steps ({future_steps}). "
            "Dataset memiliki {} baris. Butuh setidaknya {} baris.".format(
                len(data), sequence_length + future_steps
            )
        )
    return np.array(X), np.array(y)

def train_and_save_model_for_stock(stock_name, df_stock_raw, feature_columns, target_column,
                                   sequence_length, future_prediction_days, models_dir, scalers_dir,
                                   dates_to_exclude_str_list, plots_dir):
    print(f"\n--- Memproses Saham: {stock_name} ---")
    df_processed = df_stock_raw.copy()
    if dates_to_exclude_str_list:
        exclude_timestamps = pd.to_datetime(dates_to_exclude_str_list, errors='coerce').dropna()
        if not exclude_timestamps.empty:
            rows_before_exclusion = len(df_processed)
            if isinstance(df_processed.index, pd.DatetimeIndex):
                df_processed = df_processed[~df_processed.index.isin(exclude_timestamps)]
                rows_after_exclusion = len(df_processed)
                if rows_before_exclusion > rows_after_exclusion:
                    print(f"   {rows_before_exclusion - rows_after_exclusion} baris dikecualikan untuk {stock_name}.")
    if 'Vol.' in df_processed.columns and 'Vol.' in feature_columns:
        df_processed['Vol.'] = df_processed['Vol.'].apply(convert_volume_to_numeric)
    missing_cols = [col for col in feature_columns if col not in df_processed.columns]
    if missing_cols:
        print(f"Error: Kolom fitur berikut tidak ditemukan di data {stock_name}: {missing_cols}")
        return
    df_features = df_processed[feature_columns].copy()
    original_len = len(df_features)
    df_features.dropna(inplace=True)
    if df_features.empty or len(df_features) < sequence_length + future_prediction_days + 20:
        print(f"Tidak cukup data valid untuk {stock_name}. Melewati.")
        return
    print(f"Total data poin valid untuk {stock_name}: {len(df_features)}")
    print(f"Fitur yang digunakan: {feature_columns}")
    print(f"Target prediksi: {target_column}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features.values)
    safe_stock_name = stock_name.replace(' ', '_').replace('.', '').replace('/', '_').replace('\\', '_')
    scaler_filename = os.path.join(scalers_dir, f"scaler_cnn_bilstm_{safe_stock_name}.gz")
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler untuk {stock_name} disimpan di {scaler_filename}")
    try:
        target_column_index = feature_columns.index(target_column)
    except ValueError:
        print(f"Error: Kolom target '{target_column}' tidak ditemukan di: {feature_columns}")
        return
    try:
        X, y = create_sequences_multi_feature(scaled_data, sequence_length, future_prediction_days, target_column_index)
    except ValueError as e:
        print(f"Error saat membuat sekuens untuk {stock_name}: {e}")
        return
    num_features = scaled_data.shape[1]
    X = np.reshape(X, (X.shape[0], X.shape[1], num_features))
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Error: Data latih atau uji kosong untuk {stock_name}. Melewati.")
        return
    print(f"Jumlah data latih untuk {stock_name}: {len(X_train)}")
    print(f"Jumlah data uji untuk {stock_name}: {len(X_test)}")
    print(f"Bentuk X_train: {X_train.shape}, Bentuk y_train: {y_train.shape}")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(sequence_length, num_features), padding='same'),
        Bidirectional(LSTM(units=128, return_sequences=True)),
        Dropout(0.40),
        Bidirectional(LSTM(units=64, return_sequences=False)),
        Dropout(0.30),
        Dense(units=future_prediction_days)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    print(f"Melatih model CNN-BiLSTM untuk {stock_name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    model_filename = os.path.join(models_dir, f"model_cnn_bilstm_{safe_stock_name}.h5")
    model.save(model_filename)
    print(f"Model untuk {stock_name} disimpan di {model_filename}")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss pada data uji (MSE) untuk {stock_name}: {test_loss:.6f}")
    print(f"Membuat grafik untuk {stock_name}...")
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training & Validation Loss - {stock_name} (CNN-BiLSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plot_loss_path = os.path.join(plots_dir, f"{safe_stock_name}_cnn_bilstm_loss.png")
    plt.savefig(plot_loss_path)
    plt.close()
    print(f"Grafik loss disimpan di: {plot_loss_path}")
    train_predictions_scaled = model.predict(X_train, verbose=0)
    dummy_y_train_actual = np.zeros((len(y_train), num_features))
    dummy_y_train_actual[:, target_column_index] = y_train.reshape(-1)
    y_train_inversed = scaler.inverse_transform(dummy_y_train_actual)[:, target_column_index]
    dummy_train_pred = np.zeros((len(train_predictions_scaled), num_features))
    dummy_train_pred[:, target_column_index] = train_predictions_scaled.reshape(-1)
    train_predictions_inversed = scaler.inverse_transform(dummy_train_pred)[:, target_column_index]
    plt.figure(figsize=(14, 7))
    plt.plot(y_train_inversed, label='Actual Train Prices', color='dodgerblue')
    plt.plot(train_predictions_inversed, label='Predicted Train Prices', color='darkorange', linestyle='--')
    plt.title(f'Training Data: Actual vs. Predicted - {stock_name} (CNN-BiLSTM)')
    plt.xlabel(f'Time Steps (Training Set - {len(y_train_inversed)} poin)')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plot_train_path = os.path.join(plots_dir, f"{safe_stock_name}_cnn_bilstm_train_pred.png")
    plt.savefig(plot_train_path)
    plt.close()
    print(f"Grafik perbandingan data latih disimpan di: {plot_train_path}")
    test_predictions_scaled = model.predict(X_test, verbose=0)
    dummy_y_test_actual = np.zeros((len(y_test), num_features))
    dummy_y_test_actual[:, target_column_index] = y_test.reshape(-1)
    y_test_inversed = scaler.inverse_transform(dummy_y_test_actual)[:, target_column_index]
    dummy_test_pred = np.zeros((len(test_predictions_scaled), num_features))
    dummy_test_pred[:, target_column_index] = test_predictions_scaled.reshape(-1)
    test_predictions_inversed = scaler.inverse_transform(dummy_test_pred)[:, target_column_index]
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inversed, label='Actual Test Prices', color='green')
    plt.plot(test_predictions_inversed, label='Predicted Test Prices', color='red', linestyle='--')
    plt.title(f'Test Data: Actual vs. Predicted - {stock_name} (CNN-BiLSTM)')
    plt.xlabel(f'Time Steps (Test Set - {len(y_test_inversed)} poin)')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plot_test_path = os.path.join(plots_dir, f"{safe_stock_name}_cnn_bilstm_test_pred.png")
    plt.savefig(plot_test_path)
    plt.close()
    print(f"Grafik perbandingan data uji disimpan di: {plot_test_path}")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    MODELS_DIR = os.path.join(BASE_DIR, 'trained_models_mf')
    SCALERS_DIR = os.path.join(BASE_DIR, 'trained_scalers_mf')
    PLOTS_DIR = os.path.join(BASE_DIR, 'training_plots_mf')
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SCALERS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Price', 'Vol.', 'Change %']
    TARGET_COLUMN = 'Price'
    SEQUENCE_LENGTH = 90
    FUTURE_PREDICTION_DAYS = 1
    DATES_TO_EXCLUDE = []
    train_option = input("Latih untuk model spesifik ? [y/n]: ").strip().lower()
    files_to_process = []
    stock_names_to_process = []
    if train_option == 'y':
        specific_stock_name = input("Masukkan nama saham yang ingin dilatih (misalnya 'Apple', tanpa '_History.csv'): ").strip()
        specific_csv_file = os.path.join(DATASET_DIR, f"{specific_stock_name}_History.csv")
        if os.path.exists(specific_csv_file):
            files_to_process.append(specific_csv_file)
            stock_names_to_process.append(specific_stock_name)
        else:
            print(f"Error: File CSV '{specific_csv_file}' untuk saham '{specific_stock_name}' tidak ditemukan.")
            return
    elif train_option == 'n':
        all_csv_files = glob.glob(os.path.join(DATASET_DIR, '*_History.csv'))
        if not all_csv_files:
            print(f"Error: Tidak ada file CSV yang ditemukan di direktori '{DATASET_DIR}' dengan pola '*_History.csv'.")
            return
        files_to_process = all_csv_files
        for f_path in files_to_process:
            filename = os.path.basename(f_path)
            stock_name = filename.replace('_History.csv', '').replace('_history.csv', '')
            stock_names_to_process.append(stock_name)
    else:
        print("Pilihan tidak valid. Harap ketik 'y' atau 'n'.")
        return
    if not files_to_process:
        print("Tidak ada file yang akan diproses.")
        return
    print(f"\nSaham yang akan diproses: {stock_names_to_process}")
    processed_files_count = 0
    for csv_file_path, stock_name_from_file in zip(files_to_process, stock_names_to_process):
        try:
            print(f"\nMemuat data dari: {csv_file_path} untuk saham: {stock_name_from_file}")
            df_raw = pd.read_csv(csv_file_path)
            if 'Date' in df_raw.columns and 'TIMESTAMP' not in df_raw.columns:
                df_raw.rename(columns={'Date': 'TIMESTAMP'}, inplace=True)
            if 'TIMESTAMP' not in df_raw.columns:
                print(f"Peringatan: Kolom 'TIMESTAMP' atau 'Date' tidak ditemukan di {os.path.basename(csv_file_path)}. Melewati.")
                continue
            df_raw['TIMESTAMP'] = pd.to_datetime(df_raw['TIMESTAMP'], errors='coerce')
            df_raw.dropna(subset=['TIMESTAMP'], inplace=True)
            df_raw.set_index('TIMESTAMP', inplace=True)
            df_raw.sort_index(inplace=True)
            if 'Change %' in FEATURE_COLUMNS and 'Change %' in df_raw.columns:
                 print(f"   Memproses kolom 'Change %' untuk {stock_name_from_file}...")
                 df_raw['Change %'] = df_raw['Change %'].apply(convert_change_pct_to_numeric)
            train_and_save_model_for_stock(
                stock_name_from_file,
                df_raw,
                FEATURE_COLUMNS,
                TARGET_COLUMN,
                SEQUENCE_LENGTH,
                FUTURE_PREDICTION_DAYS,
                MODELS_DIR,
                SCALERS_DIR,
                DATES_TO_EXCLUDE,
                PLOTS_DIR
            )
            processed_files_count += 1
        except Exception as e:
            print(f"Error saat memproses file {csv_file_path}: {e}")
            continue
    if processed_files_count > 0:
        print(f"\n--- Proses Pelatihan CNN-BiLSTM Selesai untuk {processed_files_count} Saham ---")
    else:
        print("\n--- Tidak ada saham yang diproses. ---")

if __name__ == '__main__':
    main()