import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Função para calcular o MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Diretório base do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "PJME_hourly.csv")

# 1. Carregar e explorar os dados
df = pd.read_csv(DATA_FILE, parse_dates=["Datetime"])
df = df.set_index("Datetime")

# Verificar o formato do índice
print("Índice do DataFrame:", df.index.dtype)
print("Primeiras linhas:\n", df.head())

# Verificar valores ausentes e estatísticas descritivas
print("\nValores NaN no dataset bruto:\n", df.isna().sum())
print("\nEstatísticas descritivas do dataset bruto:\n", df.describe())

# Verificar outliers (valores fora de 3 desvios padrão)
mean_mw = df["PJME_MW"].mean()
std_mw = df["PJME_MW"].std()
outliers = df[(df["PJME_MW"] < mean_mw - 3 * std_mw) | (df["PJME_MW"] > mean_mw + 3 * std_mw)]
print("Número de outliers (fora de 3 desvios padrão):", len(outliers))

# Remover outliers (substituir por NaN e interpolar)
df["PJME_MW"] = np.where(
    (df["PJME_MW"] < mean_mw - 3 * std_mw) | (df["PJME_MW"] > mean_mw + 3 * std_mw),
    np.nan,
    df["PJME_MW"]
)
df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear")
print("\nEstatísticas descritivas após remover outliers:\n", df.describe())

# Verificar valores inconsistentes (ex.: negativos)
if (df["PJME_MW"] < 0).any():
    print("Valores negativos encontrados! Substituindo por interpolação...")
    df["PJME_MW"] = np.where(df["PJME_MW"] < 0, np.nan, df["PJME_MW"])
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear")

# EDA: Média diária ao longo de 1 ano (8760 horas = 365 dias)
df_daily = df["PJME_MW"].resample("D").mean()
plt.figure(figsize=(15, 7))
plt.plot(df_daily.index[:365], df_daily[:365], label="Daily Average Consumption (MW)")
plt.title("Daily Average Energy Consumption (First Year)")
plt.xlabel("Date")
plt.ylabel("MW")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_plots.png"))
plt.close()

# EDA: Média por hora do dia
df["hour"] = df.index.hour
hourly_avg = df.groupby("hour")["PJME_MW"].mean()
plt.figure(figsize=(10, 5))
hourly_avg.plot(kind="bar")
plt.title("Average Consumption by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("MW")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hourly_avg.png"))
plt.close()

# EDA: Decomposição da série temporal (primeiro ano)
decomposition = seasonal_decompose(df["PJME_MW"][:8760], model="additive", period=24)
plt.figure(figsize=(15, 10))
plt.subplot(411)
plt.plot(decomposition.observed, label="Observed")
plt.title("Decomposition of Energy Consumption (First Year)")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(decomposition.trend, label="Trend", color="orange")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(decomposition.seasonal, label="Seasonal", color="green")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(decomposition.resid, label="Residual", color="red")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decomposition.png"))
plt.close()

# 2. Pré-processamento
# Features temporais
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["dayofyear"] = df.index.dayofyear
df["weekofyear"] = df.index.isocalendar().week
df["hour_dayofweek"] = df["hour"] * df["dayofweek"]

# Features cíclicas para hora
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lags (múltiplos de 24h para capturar padrões diários)
df["lag_1"] = df["PJME_MW"].shift(1)
df["lag_2"] = df["PJME_MW"].shift(2)
df["lag_3"] = df["PJME_MW"].shift(3)
df["lag_24"] = df["PJME_MW"].shift(24)
df["lag_48"] = df["PJME_MW"].shift(48)
df["lag_72"] = df["PJME_MW"].shift(72)
df["lag_168"] = df["PJME_MW"].shift(168)
df["lag_336"] = df["PJME_MW"].shift(336)
df["lag_720"] = df["PJME_MW"].shift(720)
df["rolling_mean_24"] = df["PJME_MW"].rolling(window=24).mean()

# Diferenciação (y_t - y_{t-24})
df["diff_24"] = df["PJME_MW"] - df["lag_24"]

# Verificar NaNs
print("\nValores NaN após criar lags:\n", df.isna().sum())

# Preencher NaNs com interpolação linear e, se necessário, com a média da coluna
df = df.interpolate(method="linear")
print("\nValores NaN após interpolação:\n", df.isna().sum())

# Preencher NaNs restantes com a média da coluna
for col in df.columns:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
print("\nValores NaN após preenchimento com média:\n", df.isna().sum())

# Garantir que todas as colunas sejam numéricas
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("\nTipos de dados após conversão:\n", df.dtypes)

# Divisão dos dados: prever para o primeiro ano (8760 horas)
eda_period = 8760  # 365 dias
train = df.iloc[eda_period:]  # Treinar com dados após o primeiro ano
test = df.iloc[:eda_period]   # Testar no primeiro ano (mesmo período do EDA)

# 3. Modelagem - XGBoost
features = ["hour", "hour_sin", "hour_cos", "dayofweek", "month", "dayofyear", "weekofyear", "hour_dayofweek", "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "lag_720", "rolling_mean_24", "diff_24"]
X_train = train[features]
y_train = train["PJME_MW"]
X_test = test[features]
y_test = test["PJME_MW"]

model_xgb = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_xgb.fit(X_train, y_train)

# Prever para o período completo (para métricas)
xgb_pred = model_xgb.predict(X_test)

# Converter previsões para DataFrame com índice correto
xgb_pred_df = pd.DataFrame(xgb_pred, index=X_test.index, columns=["XGBoost_Pred"])

# 4. Avaliação
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
mae_xgb = mean_absolute_error(y_test, xgb_pred)
mape_xgb = mean_absolute_percentage_error(y_test, xgb_pred)
print(f"\nXGBoost - RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, MAPE: {mape_xgb:.2f}%")

# Visualizar a distribuição do erro
errors = y_test - xgb_pred
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, edgecolor="black")
plt.title("Distribuição dos Erros (XGBoost)")
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"))
plt.close()

# Debug: Mostrar os primeiros 24 valores
print("\nFirst 24 hours of forecast (same period as EDA):")
for i in range(24):
    print(f"Hour {xgb_pred_df.index[i]}: Real={y_test.iloc[i]:.2f}, XGBoost={xgb_pred_df['XGBoost_Pred'].iloc[i]:.2f}")

# Gráfico: 1 dia (24 horas) - mesmo período do EDA
plt.figure(figsize=(15, 7))
plt.plot(y_test.index[:24], y_test[:24], label="Real", color="blue", linewidth=2, marker="o")
plt.plot(xgb_pred_df.index[:24], xgb_pred_df["XGBoost_Pred"][:24], label="XGBoost", color="green", linewidth=2, linestyle="--", marker="x")
plt.title("XGBoost Forecast vs Real (First Day of Data)")
plt.xlabel("Date")
plt.ylabel("MW")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.autoscale()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forecast_1day.png"))
plt.close()

# Salvar métricas
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"XGBoost - RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, MAPE: {mape_xgb:.2f}%\n")