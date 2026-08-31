import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import io

# ==========================================
# 0. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="인도네시아 영화 박스오피스 예측 대시보드",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 데이터 로드 및 전처리 함수
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_or_path=None):
    if file_or_path is not None:
        df = pd.read_csv(file_or_path)
    else:
        # 기본 파일이 없을 경우 예시 파일명 로드
        try:
            df = pd.read_csv("movie_cgv_data.csv")
        except:
            return None

    # 숫자 컬럼 정제 (콤마, 따옴표 제거)
    numeric_cols = ['CGV Adm', 'Net Sales (IDR)', 'Seat', 'Showtime', 'IDN ADM', 'Site']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 날짜 파싱
    if 'Release Date' in df.columns:
        df['Clean_Date'] = df['Release Date'].astype(str).str.replace('?', '').str.strip()
        df['Release_Date_Parsed'] = pd.to_datetime(df['Clean_Date'], format='%m-%d-%y', errors='coerce')
        df['Release_Month'] = df['Release_Date_Parsed'].dt.month.fillna(6).astype(int)
    else:
        df['Release_Month'] = 6

    # 파생 변수
    df['Country'] = df['Country'].fillna('OTHER').str.strip().str.upper()
    df['Distributor'] = df['Distributor'].fillna('OTHER').str.strip()
    df['Genre_Rep'] = df['GENRE (REPRESENTATIVE)'].fillna(df['GENRE (ALL)'].fillna('DRAMA')).str.split(';').str[0].str.strip()
    df['Is_MCU'] = df['REMARK'].fillna('').apply(lambda x: 1 if 'MCU' in str(x) or 'DCEU' in str(x) else 0)
    df['Is_Religious'] = df['RELIGIOUS'].fillna('N').apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    df['Is_Non_XXI'] = df['NON-XXI RELEASE'].fillna('N').apply(lambda x: 1 if str(x).upper() == 'Y' else 0)

    return df

# ==========================================
# 2. 머신러닝 예측 모델 학습 (LightGBM)
# ==========================================
@st.cache_resource
def train_models(df):
    train_df = df[df['CGV Adm'].notnull() & (df['CGV Adm'] > 0) & df['Site'].notnull()].copy()
    
    features = ['Release Year', 'Release_Month', 'Site', 'Showtime', 'Seat', 
                'Country', 'Distributor', 'Genre_Rep', 'Is_MCU', 'Is_Religious']

    # 범주형 인코딩용 카테고리화
    for c in ['Country', 'Distributor', 'Genre_Rep']:
        train_df[c] = train_df[c].astype('category')

    X = train_df[features]
    y_adm = np.log1p(train_df['CGV Adm'])
    y_sales = np.log1p(train_df['Net Sales (IDR)'].fillna(train_df['CGV Adm'] * 40000))

    # CGV 관객수 예측 모델
    model_adm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1)
    model_adm.fit(X, y_adm)

    # 매출액 예측 모델
    model_sales = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1)
    model_sales.fit(X, y_sales)

    # CGV 점유율 중앙값 (인도네시아 전체 관객 역산용)
    df_valid_share = df[df['IDN ADM'].notnull() & (df['IDN ADM'] > 0) & df['CGV Adm'].notnull() & (df['CGV Adm'] > 0)]
    median_share = (df_valid_share['CGV Adm'] / df_valid_share['IDN ADM']).median()
    if np.isnan(median_share) or median_share <= 0:
        median_share = 0.12 # 기본 CGV 점유율 약 12%

    return model_adm, model_sales, median_share, train_df

# ==========================================
# 3. 사이드바 - 데이터 및 입력 파라미터
# ==========================================
st.sidebar.title("🎬 Box Office Predictor")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 CSV 데이터셋 업로드 (선택)", type=["csv"])
df = load_and_preprocess_data(uploaded_file)

if df is None:
    st.error("데이터를 불러올 수 없습니다. CSV 파일을 업로드해 주세요.")
    st.stop()

model_adm, model_sales, median_cgv_share, train_df = train_models(df)

st.sidebar.subheader("🎯 예측 영화 속성 입력")

input_title = st.sidebar.text_input("영화 제목", "New Blockbuster 2026")
input_year = st.sidebar.number_input("개봉 연도", min_value=2012, max_value=2030, value=2026)
input_month = st.sidebar.slider("개봉 월 (1~12)", min_value=1, max_value=12, value=7)

country_list = sorted(list(train_df['Country'].unique()))
input_country = st.sidebar.selectbox("제작 국가", country_list, index=country_list.index("INDONESIA") if "INDONESIA" in country_list else 0)

genre_list = sorted(list(train_df['Genre_Rep'].unique()))
input_genre = st.sidebar.selectbox("대표 장르", genre_list, index=genre_list.index("HORROR") if "HORROR" in genre_list else 0)

dist_list = sorted(list(train_df['Distributor'].unique()))
input_dist = st.sidebar.selectbox("배급사", dist_list, index=0)

st.sidebar.markdown("##### ⚙️ 배급 규모 & 특성")
input_site = st.sidebar.slider("목표 상영관 수 (Site)", min_value=1, max_value=80, value=65)
input_showtime = st.sidebar.number_input("총 상영 횟수 (Showtime)", min_value=10, max_value=50000, value=3500)
input_seat = st.sidebar.number_input("총 배정 좌석 수 (Seat)", min_value=1000, max_value=5000000, value=500000)

input_mcu = st.sidebar.checkbox("MCU / DCEU 프랜차이즈 여부", value=False)
input_rel = st.sidebar.checkbox("종교/이슬람 테마 영화 여부", value=False)

# ==========================================
# 4. 예측 수행
# ==========================================
input_data = pd.DataFrame([{
    'Release Year': input_year,
    'Release_Month': input_month,
    'Site': input_site,
    'Showtime': input_showtime,
    'Seat': input_seat,
    'Country': input_country,
    'Distributor': input_dist,
    'Genre_Rep': input_genre,
    'Is_MCU': 1 if input_mcu else 0,
    'Is_Religious': 1 if input_rel else 0
}])

for c in ['Country', 'Distributor', 'Genre_Rep']:
    input_data[c] = input_data[c].astype('category')

pred_log_adm = model_adm.predict(input_data)[0]
pred_adm = max(0, int(np.expm1(pred_log_adm)))

pred_log_sales = model_sales.predict(input_data)[0]
pred_sales = max(0, int(np.expm1(pred_log_sales)))

# 인도네시아 전체 관객수 추정 (CGV 마켓셰어 기반)
pred_idn_adm = int(pred_adm / median_cgv_share) if median_cgv_share > 0 else pred_adm * 7

# ==========================================
# 5. 메인 대시보드 UI 레이아웃
# ==========================================
st.title(f"📊 영화 흥행 예측 리포트: <{input_title}>")
st.markdown(f"**{input_year}년 {input_month}월 개봉** | **장르**: {input_genre} | **국가**: {input_country} | **배급**: {input_dist}")
st.write("")

# KPI 요약 카드
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🎟️ 예상 CGV 관객 수", value=f"{pred_adm:,} 명", help="CGV 체인점 기준 누적 예상 관객 수")
with col2:
    st.metric(label="💰 예상 총 매출 (IDR)", value=f"Rp {pred_sales/1e9:.2f} B", help="인도네시아 루피아 (B=10억)")
with col3:
    st.metric(label="🇮🇩 인도네시아 전체 예상 관객", value=f"{pred_idn_adm:,} 명", help="전체 극장 체인(XXI, Cinepolis 포함) 합산 추정치")
with col4:
    occ_rate = (pred_adm / input_seat * 100) if input_seat > 0 else 0
    st.metric(label="💺 예상 좌석 점유율", value=f"{occ_rate:.1f} %")

st.markdown("---")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["🔮 상세 시나리오 분석", "🔍 유사 벤치마크 영화", "📈 시장 트렌드 & 인사이트"])

# ----------------------------------------------------
# TAB 1: 시나리오별 관객수 범위 & 매출 전망
# ----------------------------------------------------
with tab1:
    st.subheader("📌 시나리오별 흥행 전망 (Low / Base / High)")
    
    scenarios = pd.DataFrame({
        "구분": ["보수적 시나리오 (Low)", "기본 예상치 (Base)", "낙관적 시나리오 (High)"],
        "CGV 관객 수": [int(pred_adm * 0.75), pred_adm, int(pred_adm * 1.35)],
        "예상 매출액 (IDR)": [f"Rp {(pred_sales*0.75)/1e9:.2f} B", f"Rp {pred_sales/1e9:.2f} B", f"Rp {(pred_sales*1.35)/1e9:.2f} B"],
        "인도네시아 전체 관객": [int(pred_idn_adm * 0.75), pred_idn_adm, int(pred_idn_adm * 1.35)],
        "좌석 점유율": [f"{occ_rate*0.75:.1f}%", f"{occ_rate:.1f}%", f"{min(100.0, occ_rate*1.35):.1f}%"]
    })
    
    st.dataframe(scenarios, use_container_width=True, hide_index=True)

    # 시나리오 차트
    fig_scen = go.Figure(data=[
        go.Bar(name='CGV 관객', x=scenarios['구분'], y=scenarios['CGV 관객 수'], marker_color='#E50914'),
        go.Bar(name='인도네시아 전체 관객', x=scenarios['구분'], y=scenarios['인도네시아 전체 관객'], marker_color='#1E88E5')
    ])
    fig_scen.update_layout(barmode='group', title_text="시나리오별 관객 수 비교", template="plotly_white")
    st.plotly_chart(fig_scen, use_container_width=True)

# ----------------------------------------------------
# TAB 2: 유사 벤치마크 영화 비교
# ----------------------------------------------------
with tab2:
    st.subheader(f"🎯 동일 장르({input_genre}) 및 국가({input_country}) 과거 벤치마크")
    
    sim_df = df[(df['Country'] == input_country) & (df['Genre_Rep'] == input_genre)].copy()
    if len(sim_df) == 0:
        sim_df = df[df['Genre_Rep'] == input_genre].copy()
    
    sim_df = sim_df.sort_values(by='CGV Adm', ascending=False).head(10)
    
    if len(sim_df) > 0:
        display_cols = ['TITLE', 'Release Year', 'Distributor', 'Site', 'CGV Adm', 'IDN ADM', 'Net Sales (IDR)']
        st.dataframe(sim_df[[c for c in display_cols if c in sim_df.columns]], use_container_width=True, hide_index=True)
        
        # 벤치마크 바 차트
        fig_sim = px.bar(
            sim_df.head(7), 
            x='CGV Adm', 
            y='TITLE', 
            orientation='h',
            title="상위 유사 작품 CGV 관객 수 비교",
            color='CGV Adm',
            color_continuous_scale='Reds'
        )
        fig_sim.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.info("비교할 수 있는 유사 데이터가 충분하지 않습니다.")

# ----------------------------------------------------
# TAB 3: 시장 트렌드 & 인사이트
# ----------------------------------------------------
with tab3:
    st.subheader("📊 인도네시아 박스오피스 주요 통계")
    
    tcol1, tcol2 = st.columns(2)
    
    with tcol1:
        # 장르별 평균 CGV 관객수
        genre_grp = df.groupby('Genre_Rep')['CGV Adm'].mean().reset_index().sort_values(by='CGV Adm', ascending=False).head(8)
        fig_g = px.bar(genre_grp, x='Genre_Rep', y='CGV Adm', title="장르별 평균 CGV 관객 수", color='Genre_Rep')
        fig_g.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)
        
    with tcol2:
        # 국가별 총 관객 점유율
        cntry_grp = df.groupby('Country')['CGV Adm'].sum().reset_index().sort_values(by='CGV Adm', ascending=False).head(6)
        fig_c = px.pie(cntry_grp, values='CGV Adm', names='Country', title="제작 국가별 CGV 관객 점유율", hole=0.4)
        st.plotly_chart(fig_c, use_container_width=True)

    # 월별 계절성 분석
    month_grp = df.groupby('Release_Month')['CGV Adm'].mean().reset_index()
    fig_m = px.line(month_grp, x='Release_Month', y='CGV Adm', markers=True, title="개봉 월별 평균 관객 수 추이 (시즌 효과)")
    fig_m.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), template="plotly_white")
    st.plotly_chart(fig_m, use_container_width=True)
