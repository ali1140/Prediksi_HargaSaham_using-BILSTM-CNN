import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
from datetime import datetime, timedelta

def convert_volume_to_numeric(volume_str):
    if isinstance(volume_str, (int, float)):
        return volume_str
    if pd.isna(volume_str):
        return np.nan
    volume_str = str(volume_str).strip().upper()
    if not volume_str: return np.nan
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    for suffix, multiplier in multipliers.items():
        if volume_str.endswith(suffix):
            try: return float(volume_str[:-len(suffix)]) * multiplier
            except ValueError: return np.nan
    try: return float(volume_str)
    except ValueError: return np.nan

def convert_change_pct_to_numeric(change_str):
    if isinstance(change_str, (int, float)):
        return change_str
    if pd.isna(change_str):
        return np.nan
    change_str = str(change_str).strip()
    if not change_str: return np.nan
    try: return float(change_str.rstrip('%')) / 100.0
    except ValueError: return np.nan

def main_predict_loop():
    # --- Konfigurasi Awal ---
    NAMA_FILE_CSV_DIR = 'dataset'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_BASE_PATH = os.path.join(BASE_DIR, NAMA_FILE_CSV_DIR)

    MODELS_DIR = os.path.join(BASE_DIR, 'trained_models_mf')
    SCALERS_DIR = os.path.join(BASE_DIR, 'trained_scalers_mf')

    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Price', 'Vol.', 'Change %']
    TARGET_COLUMN = 'Price'
    sequence_length = 90

    print("Selamat datang di program prediksi harga saham LSTM.")
    print("Model dan scaler diasumsikan dilatih dengan fitur:", FEATURE_COLUMNS)
    print(f"Panjang sekuens (sequence_length) yang digunakan: {sequence_length}")
    print("-" * 50)

    while True:
        print("\nMasukkan detail prediksi baru atau ketik 'exit' pada nama saham untuk keluar.")
        stock_name_input_raw = input("Masukkan nama saham (misalnya 'Apple', 'NVIDIA'): ").strip()
        if stock_name_input_raw.lower() == 'exit':
            print("Terima kasih! Keluar dari program.")
            break

        tanggal_target_str = input(f"Masukkan tanggal target prediksi untuk {stock_name_input_raw} (format YYYY/MM/DD atau YYYY-MM-DD): ").strip()

        csv_file_name = f"{stock_name_input_raw}_History.csv"
        csv_file_path = os.path.join(DATASET_BASE_PATH, csv_file_name)

        safe_stock_name_file = stock_name_input_raw.replace(' ', '_').replace('.', '').replace('/', '_').replace('\\', '_')
        model_path = os.path.join(MODELS_DIR, f"model_cnn_bilstm_{safe_stock_name_file}.h5")
        scaler_path = os.path.join(SCALERS_DIR, f"scaler_cnn_bilstm_{safe_stock_name_file}.gz")

        if not os.path.exists(csv_file_path):
            print(f"❌ Error: File data CSV '{csv_file_path}' tidak ditemukan. Silakan coba lagi.")
            print("-" * 50)
            continue
        if not os.path.exists(model_path):
            print(f"❌ Error: File model '{model_path}' tidak ditemukan. Pastikan model telah dilatih.")
            print("-" * 50)
            continue
        if not os.path.exists(scaler_path):
            print(f"❌ Error: File scaler '{scaler_path}' tidak ditemukan. Pastikan model telah dilatih.")
            print("-" * 50)
            continue

        try:
            print(f"🔄 Memuat model dari: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model berhasil dimuat.")
            print(f"🔄 Memuat scaler dari: {scaler_path}")
            scaler = joblib.load(scaler_path)
            print("✅ Scaler berhasil dimuat.")
        except Exception as e:
            print(f"❌ Error saat memuat model atau scaler: {e}")
            print("-" * 50)
            continue

        try:
            df_full = pd.read_csv(csv_file_path)

            if 'Date' in df_full.columns and 'TIMESTAMP' not in df_full.columns:
                df_full.rename(columns={'Date': 'TIMESTAMP'}, inplace=True)
            if 'TIMESTAMP' not in df_full.columns:
                print(f"❌ Error: Kolom 'TIMESTAMP' atau 'Date' tidak ditemukan di file CSV '{csv_file_name}'.")
                print("-" * 50)
                continue

            df_full['TIMESTAMP'] = pd.to_datetime(df_full['TIMESTAMP'], errors='coerce')
            df_full.dropna(subset=['TIMESTAMP'], inplace=True)
            df_full.set_index('TIMESTAMP', inplace=True)
            df_full.sort_index(inplace=True)

            if 'Vol.' in df_full.columns:
                df_full['Vol.'] = df_full['Vol.'].apply(convert_volume_to_numeric)
            if 'Change %' in FEATURE_COLUMNS and 'Change %' in df_full.columns:
                df_full['Change %'] = df_full['Change %'].apply(convert_change_pct_to_numeric)
            elif 'Change %' in FEATURE_COLUMNS and 'Change %' not in df_full.columns:
                 print(f"❌ Error: Fitur 'Change %' dibutuhkan tetapi tidak ditemukan di CSV '{csv_file_name}'.")
                 print("-" * 50)
                 continue

            missing_cols = [col for col in FEATURE_COLUMNS if col not in df_full.columns]
            if missing_cols:
                print(f"❌ Error: Kolom fitur berikut tidak ditemukan di CSV setelah preprocessing: {missing_cols}")
                print(f"   Kolom yang tersedia: {df_full.columns.tolist()}")
                print("-" * 50)
                continue

            df_stock_features = df_full[FEATURE_COLUMNS].copy()
            df_stock_features.dropna(inplace=True)

            if df_stock_features.empty:
                print(f"❌ Error: Tidak ada data valid setelah preprocessing untuk saham '{stock_name_input_raw}'.")
                print("-" * 50)
                continue

            try:
                target_column_index_in_features = FEATURE_COLUMNS.index(TARGET_COLUMN)
                open_idx = FEATURE_COLUMNS.index('Open')
                high_idx = FEATURE_COLUMNS.index('High')
                low_idx = FEATURE_COLUMNS.index('Low')
            except ValueError as e:
                print(f"❌ Error: Salah satu kolom 'Open', 'High', 'Low', atau '{TARGET_COLUMN}' tidak ada dalam FEATURE_COLUMNS: {e}")
                print("-" * 50)
                continue

        except Exception as e:
            print(f"❌ Error saat memuat atau memproses data saham: {e}")
            print("-" * 50)
            continue

        try:
            target_date_dt = pd.to_datetime(tanggal_target_str)
        except ValueError:
            print(f"❌ Error: Format tanggal target '{tanggal_target_str}' tidak valid.")
            print("-" * 50)
            continue

        if df_stock_features.index.tz is not None and target_date_dt.tz is None:
            target_date_dt = target_date_dt.tz_localize(df_stock_features.index.tz)
        elif df_stock_features.index.tz is None and target_date_dt.tz is not None:
             target_date_dt = target_date_dt.tz_convert(None).tz_localize(None)

        last_historical_date_in_features = df_stock_features.index.max()
        actual_price_on_target_date = None
        if target_date_dt in df_stock_features.index:
            actual_price_on_target_date = df_stock_features.loc[target_date_dt, TARGET_COLUMN]

        predicted_price_final = None
        num_features = len(FEATURE_COLUMNS)

        if target_date_dt <= last_historical_date_in_features:
            print(f"ℹ️ Melakukan prediksi untuk tanggal historis: {target_date_dt.strftime('%Y-%m-%d')}")
            data_up_to_target_exclusive = df_stock_features[df_stock_features.index < target_date_dt]

            if len(data_up_to_target_exclusive) < sequence_length:
                print(f"❌ Error: Tidak cukup data historis ({len(data_up_to_target_exclusive)}) sebelum {target_date_dt.strftime('%Y-%m-%d')} (butuh {sequence_length}).")
                print("-" * 50)
                continue

            input_sequence_raw_features = data_up_to_target_exclusive.iloc[-sequence_length:].values
            input_sequence_scaled = scaler.transform(input_sequence_raw_features)
            current_batch = input_sequence_scaled.reshape(1, sequence_length, num_features)
            
            predicted_target_scaled_tensor = model.predict(current_batch, verbose=0)
            predicted_target_scaled_val = predicted_target_scaled_tensor[0][0]

            dummy_array_for_inverse = np.zeros((1, num_features))
            dummy_array_for_inverse[0, target_column_index_in_features] = predicted_target_scaled_val
            predicted_price_final = scaler.inverse_transform(dummy_array_for_inverse)[0, target_column_index_in_features]

        else:
            print(f"ℹ️ Tanggal target {target_date_dt.strftime('%Y-%m-%d')} berada di masa depan.")
            print(f"   Data historis terakhir pada: {last_historical_date_in_features.strftime('%Y-%m-%d')}")
            print("   Melakukan prediksi iteratif. PERINGATAN: Akurasi dapat menurun drastis!")

            num_steps_to_predict = (target_date_dt.normalize() - last_historical_date_in_features.normalize()).days
            
            if len(df_stock_features) < sequence_length:
                 print(f"❌ Error: Tidak cukup data historis ({len(df_stock_features)}) untuk memulai prediksi iteratif (butuh {sequence_length}).")
                 print("-" * 50)
                 continue

            current_raw_sequence_features = df_stock_features.iloc[-sequence_length:].values
            current_full_sequence_scaled = scaler.transform(current_raw_sequence_features)
            
            print(f"   Memulai prediksi iteratif untuk {num_steps_to_predict} langkah...")
            
            for step in range(num_steps_to_predict):
                batch_to_predict = current_full_sequence_scaled[-sequence_length:].reshape(1, sequence_length, num_features)

                predicted_target_next_day_scaled_val = model.predict(batch_to_predict, verbose=0)[0][0]
                
                new_feature_row_scaled = current_full_sequence_scaled[-1, :].copy() 
                new_feature_row_scaled[target_column_index_in_features] = predicted_target_next_day_scaled_val
                new_feature_row_scaled[open_idx] = predicted_target_next_day_scaled_val
                new_feature_row_scaled[high_idx] = predicted_target_next_day_scaled_val
                new_feature_row_scaled[low_idx] = predicted_target_next_day_scaled_val

                current_full_sequence_scaled = np.vstack([current_full_sequence_scaled[1:], new_feature_row_scaled])
                
                if (step + 1) % 10 == 0 or step == num_steps_to_predict - 1:
                     print(f"    Langkah iterasi {step+1}/{num_steps_to_predict} selesai.")

            final_predicted_scaled_val = current_full_sequence_scaled[-1, target_column_index_in_features]
            dummy_inv = np.zeros((1, num_features))
            dummy_inv[0, target_column_index_in_features] = final_predicted_scaled_val
            predicted_price_final = scaler.inverse_transform(dummy_inv)[0, target_column_index_in_features]

        if predicted_price_final is not None:
            print(f"\n🎯 **Hasil Prediksi untuk Saham: {stock_name_input_raw}**")
            print(f"   Tanggal Target Prediksi: {target_date_dt.strftime('%Y-%m-%d')}")
            if actual_price_on_target_date is not None:
                print(f"   Harga Aktual Saham ({TARGET_COLUMN}) : {actual_price_on_target_date:.2f}")
            elif target_date_dt <= last_historical_date_in_features:
                print(f"   Harga Aktual Saham ({TARGET_COLUMN}) : Tidak tersedia (kemungkinan hari non-perdagangan)")
            print(f"   Harga Prediksi Saham ({TARGET_COLUMN}): {predicted_price_final:.2f}")
            if actual_price_on_target_date is not None:
                selisih = predicted_price_final - actual_price_on_target_date
                persentase_selisih = (selisih / actual_price_on_target_date) * 100 if actual_price_on_target_date != 0 else float('inf')
                print(f"   Selisih (Prediksi - Aktual) : {selisih:.2f} ({persentase_selisih:.2f}%)")
        else:
            print("   Gagal mendapatkan hasil prediksi.")

        print("-" * 50)

if __name__ == '__main__':
    main_predict_loop()