# ============================================================
# ConvLSTM Gibrid Model - Gidrologik Anomaliya Aniqlash
# Muallif: Nasridinov Rustamjon, TATU
# Dataset: CA-discharge (Marti et al., 2023)
# Stansiyalar: Chatkal, Karadarya, Qashqadarya, Tupalang
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    ConvLSTM2D, Flatten, Reshape, Conv1D, MaxPooling1D,
    TimeDistributed, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

print(f"TensorFlow version: {tf.__version__}")

# ============================================================
# 1. MA'LUMOT YUKLASH
# ============================================================

# Stansiyalar
STATIONS = {
    'Chatkal': 'Chatkal_Chirchik_16279.csv',
    'Karadarya': 'Karadarya_SyrDarya_16938.csv',
    'Qashqadarya': 'Qashqadarya_17231.csv',
    'Tupalang': 'Tupalang_Surkhandarya_17194.csv'
}

def load_station(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.dropna(subset=['discharge_m3s'])
    df = df.sort_values('date').reset_index(drop=True)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

# ============================================================
# 2. MA'LUMOT TAYYORLASH
# ============================================================

def create_sequences(data, seq_len=24, pred_len=3):
    """
    Vaqt qatori dan ketma-ket oynalar yaratish.
    seq_len: kirish uzunligi (oy)
    pred_len: bashorat uzunligi (oy)
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def prepare_data(df, seq_len=24, pred_len=3, test_ratio=0.2):
    q = df['discharge_m3s'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    q_scaled = scaler.fit_transform(q).flatten()

    X, y = create_sequences(q_scaled, seq_len, pred_len)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM uchun shape: (samples, timesteps, features)
    X_train = X_train.reshape(-1, seq_len, 1)
    X_test = X_test.reshape(-1, seq_len, 1)

    return X_train, X_test, y_train, y_test, scaler

# ============================================================
# 3. MODELLAR
# ============================================================

def build_lstm(seq_len=24, pred_len=3):
    """Oddiy LSTM - taqqoslash uchun baseline"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1),
             kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(pred_len)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_conv_lstm(seq_len=24, pred_len=3):
    """
    Gibrid Conv1D + LSTM model.
    Conv1D - mahalliy chastota xususiyatlarini ajratadi (Wavelet o'rniga)
    LSTM   - vaqt bog'liqliklarini o'rganadi
    Bu ConvLSTM arxitekturasining 1D vaqt qatori versiyasi.
    """
    inputs = Input(shape=(seq_len, 1))

    # Konvolyutsion qatlam - mavsumiy pattern aniqlash
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # LSTM qatlam - uzoq muddatli bog'liqlik
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)

    # Chiqish
    outputs = Dense(pred_len, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_wavelet_lstm(seq_len=24, pred_len=3):
    """
    Wavelet + LSTM gibrid model.
    Turli o'lchovdagi konvolyutsion filtrlar Wavelet decomposition ni taqlid qiladi.
    """
    inputs = Input(shape=(seq_len, 1))

    # Turli miqyosli filtrlar (Wavelet multi-resolution analogi)
    x1 = Conv1D(32, kernel_size=2, activation='relu', padding='same')(inputs)  # yuqori chastota
    x2 = Conv1D(32, kernel_size=6, activation='relu', padding='same')(inputs)  # o'rta chastota
    x3 = Conv1D(32, kernel_size=12, activation='relu', padding='same')(inputs) # past chastota

    # Birlashtirish
    from tensorflow.keras.layers import Concatenate
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # BiLSTM - ikki tomonlama kontekst
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = BatchNormalization()(x)

    outputs = Dense(pred_len)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

# ============================================================
# 4. MODEL O'QITISH VA BAHOLASH
# ============================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, monitor='val_loss')
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    return history

def evaluate_model(model, X_test, y_test, scaler, pred_len=3):
    y_pred = model.predict(X_test, verbose=0)

    # Inverse transform (faqat birinchi bashorat qadami uchun)
    y_test_inv = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred[:, 0].reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    nse = 1 - np.sum((y_test_inv - y_pred_inv)**2) / np.sum((y_test_inv - y_test_inv.mean())**2)

    return rmse, mae, nse, y_test_inv, y_pred_inv

# ============================================================
# 5. ANOMALIYA ANIQLASH
# ============================================================

def detect_anomalies(y_true, y_pred, threshold=2.0):
    """
    Bashorat xatosi asosida anomaliya aniqlash.
    Katta xato = anomal hodisa.
    """
    errors = np.abs(y_true - y_pred)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    z_scores = (errors - mean_err) / (std_err + 1e-10)
    anomalies = z_scores > threshold
    return anomalies, z_scores

# ============================================================
# 6. ASOSIY JARAYON
# ============================================================

SEQ_LEN = 24   # 24 oylik kirish
PRED_LEN = 3   # 3 oylik bashorat

results = {}
all_histories = {}

for station_name, filename in STATIONS.items():
    print(f"\n{'='*50}")
    print(f"Stansiya: {station_name}")
    print('='*50)

    df = load_station(filename)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, SEQ_LEN, PRED_LEN)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    station_results = {}
    station_histories = {}

    for model_name, build_fn in [
        ('LSTM', build_lstm),
        ('Conv-LSTM', build_conv_lstm),
        ('Wavelet-LSTM', build_wavelet_lstm)
    ]:
        print(f"  {model_name} o'qitilmoqda...", end=' ')
        model = build_fn(SEQ_LEN, PRED_LEN)
        history = train_model(model, X_train, y_train, X_test, y_test)
        rmse, mae, nse, y_true, y_pred = evaluate_model(model, X_test, y_test, scaler)
        anomalies, z_scores = detect_anomalies(y_true, y_pred)

        station_results[model_name] = {
            'rmse': rmse, 'mae': mae, 'nse': nse,
            'y_true': y_true, 'y_pred': y_pred,
            'anomalies': anomalies, 'z_scores': z_scores,
            'n_anomalies': anomalies.sum()
        }
        station_histories[model_name] = history
        print(f"RMSE={rmse:.2f}, MAE={mae:.2f}, NSE={nse:.3f}, Anomaliya={anomalies.sum()}")

    results[station_name] = station_results
    all_histories[station_name] = station_histories

# ============================================================
# 7. NATIJALAR JADVALI
# ============================================================

print("\n" + "="*70)
print("YAKUNIY NATIJALAR JADVALI")
print("="*70)
print(f"{'Stansiya':<15} {'Model':<15} {'RMSE':>8} {'MAE':>8} {'NSE':>8} {'Anomaliya':>10}")
print("-"*70)

for station, models in results.items():
    for model_name, metrics in models.items():
        print(f"{station:<15} {model_name:<15} "
              f"{metrics['rmse']:>8.2f} {metrics['mae']:>8.2f} "
              f"{metrics['nse']:>8.3f} {metrics['n_anomalies']:>10}")
    print("-"*70)

# ============================================================
# 8. VIZUALIZATSIYA
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.patch.set_facecolor('white')
colors = {'LSTM': '#1a6faf', 'Conv-LSTM': '#c1392b', 'Wavelet-LSTM': '#16a085'}

for row_idx, (station, models) in enumerate(results.items()):
    for col_idx, (model_name, metrics) in enumerate(models.items()):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor('white')

        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        anomalies = metrics['anomalies']

        ax.plot(y_true, color='gray', linewidth=1, alpha=0.7, label='Haqiqiy')
        ax.plot(y_pred, color=colors[model_name], linewidth=1.5, label='Bashorat')
        ax.scatter(np.where(anomalies)[0], y_true[anomalies],
                   color='red', s=20, zorder=5, label=f'Anomaliya ({anomalies.sum()})')

        ax.set_title(f'{station} | {model_name}\nNSE={metrics["nse"]:.3f} RMSE={metrics["rmse"]:.1f}',
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('Q (m³/s)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

axes[-1][0].set_xlabel('Test indeksi', fontsize=9)
axes[-1][1].set_xlabel('Test indeksi', fontsize=9)
axes[-1][2].set_xlabel('Test indeksi', fontsize=9)

fig.suptitle("O'zbekiston daryolari: LSTM, Conv-LSTM va Wavelet-LSTM\nBashorat natijalari va anomaliya aniqlash",
             fontsize=13, fontweight='bold')
plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('model_results.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("\nmodel_results.png saqlandi.")

# NSE taqqoslash jadvali
fig2, ax = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('white')
ax.set_facecolor('white')

station_names = list(results.keys())
x = np.arange(len(station_names))
width = 0.25

for i, model_name in enumerate(['LSTM', 'Conv-LSTM', 'Wavelet-LSTM']):
    nse_values = [results[s][model_name]['nse'] for s in station_names]
    bars = ax.bar(x + i*width, nse_values, width, label=model_name,
                  color=list(colors.values())[i], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, nse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Stansiya', fontsize=11)
ax.set_ylabel('NSE (Nash-Sutcliffe)', fontsize=11)
ax.set_title("Model samaradorligi taqqoslash (NSE ko'rsatkichi)\nYuqori = yaxshiroq",
             fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(station_names, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='NSE=0.7 (yaxshi daraja)')
ax.grid(True, alpha=0.3, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('nse_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("nse_comparison.png saqlandi.")

print("\nBarcha ishlar tugadi.")
