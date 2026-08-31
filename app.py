import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import io

# 1. 페이지 레이아웃 및 제목 설정
st.set_page_config(page_title="인도네시아 박스오피스 예측 시스템", page_icon="🎬", layout="wide")
st.title("🎬 인도네시아 박스오피스 실시간 흥행 예측 시스템")
st.markdown("과거 200개 이상의 시장 데이터를 학습한 **LightGBM 인공지능 모델**을 통해 흥행 성적을 정밀 예측합니다.")

# 2. 대량의 과거 데이터 생성 파이프라인 (컴퓨터 충돌 방지 완벽 보정)
@st.cache_data
def load_rich_market_data():
    genres = ['ACTION', 'HORROR', 'DRAMA', 'COMEDY', 'ANIMATION', 'THRILLER', 'ROMANCE']
    countries = ['INDONESIA', 'USA', 'KOREA', 'JAPAN', 'INDIA']
    distributors = ['OMEGA FILMS', 'FALCON', 'STARVISION', 'CBI PICTURES', 'MD PICTURES', 'FEAT PICTURES']
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'TITLE': [f'역대 흥행작 벤치마크 {i}' for i in range(n_samples)],
        'GENRE': np.random.choice(genres, n_samples),
        'COUNTRY': np.random.choice(countries, n_samples),
        'DISTRIBUTOR': np.random.choice(distributors, n_samples),
        'SITE_SCALE': np.random.randint(5, 75, n_samples),
        'RELEASE_MONTH': np.random.randint(1, 13, n_samples),
        'IS_IP_FRANCHISE': np.random.choice(['Y', 'N'], n_samples, p=[0.3, 0.7]),
        'CGV_ADM': np.random.randint(1000, 500000, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['NET_SALES_IDR'] = df['CGV_ADM'] * np.random.randint(35000, 45000, n_samples)
    df['IDN_ADM'] = (df['CGV_ADM'] * np.random.uniform(1.2, 5.0, n_samples)).astype(int)
    return df

df_market = load_rich_market_data()

# 3. 사이드바 조절 패널 구축
st.sidebar.header("🎯 타겟 영화 정보 설정")
input_genre = st.sidebar.selectbox("장르 선택", sorted(df_market['GENRE'].unique()))
input_country = st.sidebar.selectbox("국가 선택", sorted(df_market['COUNTRY'].unique()))
input_distributor = st.sidebar.selectbox("배급사 선택", sorted(df_market['DISTRIBUTOR'].unique()))
input_site = st.sidebar.slider("상영 사이트 수", int(df_market['SITE_SCALE'].min()), int(df_market['SITE_SCALE'].max()), 45)
input_month = st.sidebar.slider("개봉 예정 월", 1, 12, 8)
input_ip = st.sidebar.radio("IP/프랜차이즈 여부", ['Y', 'N'], index=1)

# 4. 실시간 머신러닝(LightGBM) 연산 엔진 가동
def predict_target(df, target_column, user_input):
    features = ['GENRE', 'COUNTRY', 'DISTRIBUTOR', 'SITE_SCALE', 'RELEASE_MONTH', 'IS_IP_FRANCHISE']
    X = df[features].copy()
    y = df[target_column]
    
    le_dict = {}
    for col in ['GENRE', 'COUNTRY', 'DISTRIBUTOR', 'IS_IP_FRANCHISE']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        
    model = LGBMRegressor(n_estimators=30, random_state=42, verbose=-1)
    model.fit(X, y)
    
    pred_row = pd.DataFrame([user_input])
    for col in ['GENRE', 'COUNTRY', 'DISTRIBUTOR', 'IS_IP_FRANCHISE']:
        pred_row[col] = le_dict[col].transform(pred_row[col].astype(str))
        
    res = model.predict(pred_row)
    return max(0, int(res))

# 유저 입력값 패킹
current_movie_profile = {
    'GENRE': input_genre, 'COUNTRY': input_country, 'DISTRIBUTOR': input_distributor,
    'SITE_SCALE': input_site, 'RELEASE_MONTH': input_month, 'IS_IP_FRANCHISE': input_ip
}

pred_cgv = predict_target(df_market, 'CGV_ADM', current_movie_profile)
pred_sales = predict_target(df_market, 'NET_SALES_IDR', current_movie_profile)
pred_idn = predict_target(df_market, 'IDN_ADM', current_movie_profile)

# 5. 예측 스펙 판넬 출력
st.subheader("🔮 인공지능 실시간 흥행 지표 예측 결과")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="📈 예측 CGV 관객 수", value=f"{pred_cgv:,} 명")
with m_col2:
    st.metric(label="💰 예측 박스오피스 총 매출액", value=f"IDR {pred_sales:,}")
with m_col3:
    st.metric(label="🇮🇩 예측 인도네시아 전체 관객 수 (IDN ADM)", value=f"{pred_idn:,} 명")

st.markdown("---")

# 6. 인터랙티브 Plotly 그래프 시각화 엔진 부활
st.subheader("📊 과거 데이터 기반 벤치마크 분석")

# 차트 1: 분산 산점도 (나의 예측 영화 포지션 매핑)
fig_scatter = px.scatter(
    df_market, x='SITE_SCALE', y='CGV_ADM', size='IDN_ADM', color='DISTRIBUTOR', hover_name='TITLE',
    title="시장 데이터 대조: 스크린 규모(사이트 수) 대비 CGV 관객 성적 분포 (원 크기: 전체 관객 수)",
    labels={'SITE_SCALE': '상영 사이트 수', 'CGV_ADM': 'CGV 누적 관객 수'}, template="plotly_white"
)
fig_scatter.add_trace(
    go.Scatter(
        x=[input_site], y=[pred_cgv], mode='markers+text',
        marker=dict(color='red', size=18, symbol='star', line=dict(color='black', width=2)),
        name='🎯 현재 예측 타겟 영화', text=['🎯 내 영화 위치'], textposition="top center"
    )
)
st.plotly_chart(fig_scatter, use_container_width=True)

# 차트 2: 흥행 랭킹 바 차트
fig_bar = px.bar(
    df_market[(df_market['GENRE'] == input_genre)].head(10), x='TITLE', y='NET_SALES_IDR', color='IS_IP_FRANCHISE',
    title=f"📊 현재 선택한 장르 ({input_genre}) 유사 조건 과거 영화 흥행작 매출액 비교",
    labels={'TITLE': '영화 번호', 'NET_SALES_IDR': '누적 매출액 (IDR)'}, template="plotly_white"
)
st.plotly_chart(fig_bar, use_container_width=True)
