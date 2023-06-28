import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('korea_data_origin\동작구청_data_csv.csv', encoding='utf-8-sig')

df['충전시작시간'] = pd.to_datetime(df['충전시작시간'])
df['충전일'] = pd.to_datetime(df['충전일'])

x_date_range = pd.date_range("2021-06-17", "2021-10-11", freq="2D")
y_range = range(0, 25, 1)

df["충전시작시간_hour"] = df["충전시작시간"].dt.hour

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(df["충전일"], df["충전시작시간_hour"])

ax.set_xticks(x_date_range)
ax.set_xticklabels(x_date_range.strftime("%m-%d"), rotation=45)

ax.set_yticks(y_range)
plt.rc('font', family='Malgun Gothic')
plt.show()

daily_usage = df.groupby('요일')['충전시간'].sum()

# 요일별 충전시간 시각화
daily_usage.plot(kind='bar', figsize=(10, 5))
plt.xlabel('요일')
plt.ylabel('충전시간')
plt.title('요일별 충전시간')
plt.rc('font', family='Malgun Gothic')
plt.show()

# 주말/평일 사용량 계산
weekend_usage = df.groupby('주말')['충전시간'].sum()

# 주말/평일 충전시간 시각화
weekend_usage.plot(kind='bar', figsize=(10, 5), xticks=[0, 1], xlabel=['평일', '주말'])
plt.ylabel('충전시간')
plt.title('주말/평일 충전시간')
plt.rc('font', family='Malgun Gothic')
plt.show()
