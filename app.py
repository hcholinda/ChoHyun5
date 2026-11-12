
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
df = pd.read_csv("data/sunspots_for_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])

# 이미지처럼 YEAR 컬럼 추가
df["YEAR"] = df["ds"].dt.year

st.subheader("📄 불러온 데이터 미리보기")
st.dataframe(df[["ds", "YEAR", "y"]].head())

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
model = Prophet(
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_mode="additive"
)

model.add_seasonality(
    name="sunspot_cycle",
    period=11,
    fourier_order=5
)

model.fit(df[["ds", "y"]])

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
future = model.make_future_dataframe(
    periods=30,
    freq="YS"
)

forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")

fig1 = model.plot(forecast)
fig1.suptitle("Prophet Forecast Plot", fontsize=16)
st.pyplot(fig1)
plt.close(fig1)

st.subheader("📊 Forecast Components")

fig2 = model.plot_components(forecast)
st.pyplot(fig2)
plt.close(fig2)

# ----------------------------------
# [5] 실제값 vs 예측값 + 예측구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    df["ds"], df["y"],
    color="blue", marker="o", label="Actual"
)

ax.plot(
    forecast["ds"], forecast["yhat"],
    color="red", linestyle="--", label="Predicted"
)

ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="pink", alpha=0.3,
    label="Prediction Interval"
)

ax.set_title("Sunspots: Actual vs. Predicted with Prediction Intervals")
ax.set_xlabel("Year")
ax.set_ylabel("Sun Activity")
ax.legend()
ax.grid(True)

st.pyplot(fig3)
plt.close(fig3)

# ----------------------------------
# [6] 잔차 분석
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

merged = pd.merge(
    df[["ds", "y"]],
    forecast[["ds", "yhat"]],
    on="ds",
    how="inner"
)

merged["residual"] = merged["y"] - merged["yhat"]

fig4, ax2 = plt.subplots(figsize=(14, 4))

ax2.plot(
    merged["ds"],
    merged["residual"],
    color="purple",
    marker="o",
    label="Residual"
)

ax2.axhline(0, color="black", linestyle="--")

ax2.set_title("Residual Analysis (Actual - Predicted)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)

st.pyplot(fig4)
plt.close(fig4)

# ----------------------------------
# [7] 잔차 통계 요약
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")

st.dataframe(merged["residual"].describe().to_frame())
