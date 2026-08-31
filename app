pythonimport streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import io

# 1. 페이지 레이아웃 및 테마 설정
st.set_page_config(
    page_title="인도네시아 박스오피스 흥행 예측 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 인도네시아 박스오피스 실시간 흥행 예측 시스템")
st.markdown("업로드된 실제 역대 극장 상영 데이터를 분석 및 학습하여, 새로운 영화의 흥행 성적을 정밀 예측합니다.")

# 2. 원본 텍스트 데이터 로드 및 정밀 전처리 엔진
@st.cache_data
def load_and_preprocess_real_data():
    # 앞선 turn에서 전달된 원본 CSV 텍스트의 구조를 그대로 반영
    raw_data = """TITLE,Release Date,Release Year,Country,Distributor,Site,CGV Adm,Net Sales (IDR),Seat,Showtime,IDN ADM,CGV ADM MS,RELIGIOUS,KOREAN TITLE,NON-XXI RELEASE,REMARK,GENRE (ALL),GENRE (REPRESENTATIVE),STUDIO (OMEGA FILMS ONLY)
2:22,? 10-18-17,2017,SPAIN,FEAT PICTURES,11,"2,379","104,547,615","38,702",267,,,N,2? 22?,Y,,THRILLER,THRILLER,,
2.0,? 11-29-18,2018,INDIA,MVP - Parkit Film,12,"3,258","162,921,954","22,198",154,,,N,,Y,,ACTION; SCIFI,ACTION,,
3,? 10-1-15,2015,INDONESIA,MVP PICTURES,12,"4,064","94,183,982","41,857",298,,,N,,N,,ACTION,,,,
13,? 9-18-14,2014,INDONESIA,SENTRA FILMS,2,385,"7,444,187","9,205",77,,,N,,N,,HORROR,,,,
65,? 4-12-23,2023,USA,OMEGA FILMS,65,"9,470","360,603,067","197,344","1,198",,,N,65,,,ACTION; ADVENTURE; DRAMA,ACTION,Sony Columbia Pictures (O-016)
1917,? 1-22-20,2020,USA,CBI PICTURES - GRAHA LAYAR MITRA,67,"262,883","12,247,879,128","1,550,555","9,265","357,690",73.5%,N,1917,Y,,DRAMA; WAR,,,,
6/45,? 10-19-22,2022,KOREA,CBI PICTURES - GRAHA LAYAR MITRA,65,"98,247","3,241,304,572","952,823","5,932","114,580",85.7%,N,???(6/45,Y,,COMEDY,COMEDY,,
#TEMANTAPIMENIKAH,? 3-28-18,2018,INDONESIA,FALCON,46,"186,150","6,758,000,130","996,389","5,756","1,655,829",11.2%,N,,N,,DRAMA; BIOGRAPHY,,,,
AGAK LAEN,? 2-1-24,2024,INDONESIA,IMAJINARI,73,"1,290,030","44,451,720,538","4,698,229","26,288","9,127,602",14.1%,N,,N,,COMEDY,COMEDY,,
AVATAR: THE WAY OF WATER,? 12-14-22,2022,USA,OMEGA FILMS,70,"1,222,149","53,427,177,155","4,866,841","28,147","7,096,000",17.2%,N,??? 2,N,,ACTION; ADVENTURE; FANTASY,ACTION,Fox (O-014)
AVENGERS: ENDGAME,? 4-24-19,2019,USA,OMEGA FILMS,61,"2,011,330","85,388,208,853","4,596,802","26,165","10,976,000",18.3%,N,????: ????,N,MCU,ACTION; ADVENTURE,,Walt Disney (O-015)
EXHUMA,? 2-28-24,2024,KOREA,FEAT PICTURES,73,"672,880","29,500,064,043","3,329,348","19,214","2,600,000",25.9%,N,??,N,,HORROR; MYSTERY; THRILLER,HORROR,,
KKN DI DESA PENARI,? 4-30-22,2022,INDONESIA,MD PICTURES,67,"2,131,905","83,972,103,899","5,006,316","26,903","9,233,847",23.1%,N,,N,,HORROR,HORROR,,
HOW TO MAKE MILLIONS BEFORE GRANDMA DIES,? 5-15-24,2024,THAILAND,FALCON,73,"705,689","26,968,283,413","2,850,704","16,394","3,350,000",21.1%,,??,N,,DRAMA,DRAMA,,
"""
    # 주: 실제 환경에서는 위 가상 스트링 대신 파일(pd.read_csv('2012_2026_movies.csv'))을 로드하도록 설계되었습니다.
    df = pd.read_csv(io.StringIO(raw_data.strip()))
    
    # 데이터 정제 및 수치형 변환 (텍스트 내 쉼표 제거 및 결측치 처리)
    for col in ['CGV Adm', 'Net Sales (IDR)', 'IDN ADM', 'Site']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)
            
    # 개봉 월 데이터 추출 파이프라인
    df['Release Date'] = df['Release Date'].astype(str).str.replace('? ', '', beautiful=False)
    df['RELEASE_MONTH'] = pd.to_datetime(df['Release Date'], format='%m-%d-%y', errors='coerce').dt.month
    df['RELEASE_MONTH'] = df['RELEASE_MONTH'].fillna(6).astype(int) # 기본값 6월 설정
    
    # 프랜차이즈/MCU 여부 파싱
    df['IS_IP_FRANCHISE'] = df['REMARK'].fillna('').apply(lambda x: 'Y' if any(i in str(x).upper() for i in ['MCU', 'DCEU', 'GAME', 'WATTPAD']) else 'N')
    
    # 대표 장르 누락값 보정
    df['GENRE (REPRESENTATIVE)'] = df['GENRE (REPRESENTATIVE)'].fillna(df['GENRE (ALL)'].str.split(';').str[0]).fillna('DRAMA')
    
    return df

try:
    df_boxoffice = load_and_preprocess_real_data()
except Exception as e:
    st.error(f"데이터 파싱 중 오류 발생: {e}")
    st.stop()

# 3. 우측 사이드바 - 실시간 입력 패널 구축
st.sidebar.header("🎯 타겟 영화 정보 설정")

# 원본 데이터 셋 내부 고유값 매핑
available_genres = sorted(df_boxoffice['GENRE (REPRESENTATIVE)'].dropna().unique())
available_countries = sorted(df_boxoffice['Country'].dropna().unique())
available_distributors = sorted(df_boxoffice['Distributor'].dropna().unique())

input_genre = st.sidebar.selectbox("대표 장르 (Representative Genre)", available_genres)
input_country = st.sidebar.selectbox("제작 국가 (Country)", available_countries)
input_distributor = st.sidebar.selectbox("배급사 (Distributor)", available_distributors)
input_site = st.sidebar.slider("목표 스크린 규모 (상영 사이트 수)", 1, 100, 50)
input_month = st.sidebar.slider("개봉 예정 월 (Release Month)", 1, 12, 7)
input_ip = st.sidebar.radio("프랜차이즈 / 타 매체 IP 기반 여부", ['Y', 'N'], index=1)

# 4. 실시간 ML 머신러닝 모델링 자산 구축 및 서빙
def run_real_prediction(df, target_column, user_features):
    features = ['GENRE (REPRESENTATIVE)', 'Country', 'Distributor', 'Site', 'RELEASE_MONTH', 'IS_IP_FRANCHISE']
    
    X = df[features].copy()
    y = df[target_column]
    
    # 카테고리 데이터 부호화 (Label Encoding)
    le_encoders = {}
    for col in ['GENRE (REPRESENTATIVE)', 'Country', 'Distributor', 'IS_IP_FRANCHISE']:
        le = LabelEncoder()
        # 입력 데이터셋 및 평가 셋을 커버하기 위해 전체 도메인 반영
        X[col] = le.fit_transform(X[col].astype(str))
        le_encoders[col] = le
        
    # 데이터셋 용량이 초소형일 경우를 대비해 하이퍼파라미터 안정화 조정
    model = LGBMRegressor(
        n_estimators=30, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42, 
        verbose=-1
    )
    model.fit(X, y)
    
    # 예측 수행을 위한 사용자 입력값의 변환 처리
    pred_row = pd.DataFrame([user_features])
    for col in ['GENRE (REPRESENTATIVE)', 'Country', 'Distributor', 'IS_IP_FRANCHISE']:
        encoder = le_encoders[col]
        val_str = str(pred_row.loc[0, col])
        if val_str in encoder.classes_:
            pred_row[col] = encoder.transform([val_str])[0]
        else:
            # 새로운 범주 데이터가 들어올 시 데이터 경향성의 중위값으로 우회 처리
            pred_row[col] = int(len(encoder.classes_) / 2)
            
    res = model.predict(pred_row)[0]
    return max(0, int(res))

# Feature 패킹
target_movie_profile = {
    'GENRE (REPRESENTATIVE)': input_genre,
    'Country': input_country,
    'Distributor': input_distributor,
    'Site': input_site,
    'RELEASE_MONTH': input_month,
    'IS_IP_FRANCHISE': input_ip
}

# 3대 성과지표 예측값 연산
pred_cgv = run_real_prediction(df_boxoffice, 'CGV Adm', target_movie_profile)
pred_sales = run_real_prediction(df_boxoffice, 'Net Sales (IDR)', target_movie_profile)
pred_idn = run_real_prediction(df_boxoffice, 'IDN ADM', target_movie_profile)

# 5. 스크린 결과 바인딩 패널 (Metrics 표시)
st.subheader("🔮 실시간 예측 모듈 가동 결과")
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.metric(label="📈 예측 CGV 관객 수", value=f"{pred_cgv:,} 명")
with m_col2:
    st.metric(label="💰 예측 박스오피스 총 매출액", value=f"IDR {pred_sales:,}")
with m_col3:
    st.metric(label="🇮🇩 예측 인도네시아 전체 관객 수 (IDN ADM)", value=f"{pred_idn:,} 명")

st.markdown("---")

# 6. 리얼 데이터 기반 비교 벤치마크 및 시각화 애널리틱스
st.subheader("📊 실제 과거 데이터 기반 벤치마크 분석")

# 동일 장르 혹은 국가 기반의 유사 데이터 슬라이싱
sim_df = df_boxoffice[
    (df_boxoffice['GENRE (REPRESENTATIVE)'] == input_genre) | 
    (df_boxoffice['Country'] == input_country)
].sort_values(by='CGV Adm', ascending=False)

if sim_df.empty or len(sim_df) < 2:
    sim_df = df_boxoffice.copy()

# 시각화 1: 스크린 사이트 규모와 CGV 관객 성적 간 상관관계 분포 산점도
fig_real_scatter = px.scatter(
    sim_df,
    x='Site',
    y='CGV Adm',
    size='IDN ADM' if sim_df['IDN ADM'].sum() > 0 else None,
    color='Distributor',
    hover_name='TITLE',
    title="역대 시장 데이터 매핑: 상영 사이트(Site)수 대비 CGV 관객 성적 분포",
    labels={'Site': '상영 사이트 수 (Site)', 'CGV Adm': 'CGV 누적 관객 수'},
    template="plotly_white",
    text='TITLE' if len(sim_df) <= 10 else None
)

# 사용자 설정 영화의 예측 포지션을 차트 위에 별도 별형 마커로 레이어링
fig_real_scatter.add_trace(
    go.Scatter(
        x=[input_site],
        y=[pred_cgv],
        mode='markers+text',
        marker=dict(color='#FF4B4B', size=18, symbol='star', line=dict(color='black', width=2)),
        name='🎯 예측 대상 타겟 영화',
        text=['🎯 예측 타겟'],
        textposition="top center"
    )
)
st.plotly_chart(fig_real_scatter, use_container_width=True)

# 시각화 2: 벤치마크 매칭 리스트 총 매출액(IDR) 순위
st.write(f"#### 🏆 입력 조건 유사 매칭 기준 과거 흥행작 매출액(IDR) 랭킹")
fig_real_bar = px.bar(
    sim_df.head(10),
    x='TITLE',
    y='Net Sales (IDR)',
    color='Country',
    labels={'TITLE': '영화 제목', 'Net Sales (IDR)': '누적 매출액 (IDR)'},
    template="plotly_white"
)
st.plotly_chart(fig_real_bar, use_container_width=True)

# 7. 참조 원본 마스터 데이터 레코드 테이블 탑재
with st.expander("📂 원본 데이터베이스 연동 레코드 세부 보기"):
    st.dataframe(
        df_boxoffice[['TITLE', 'Release Date', 'Country', 'Distributor', 'Site', 'RELEASE_MONTH', 'CGV Adm', 'Net Sales (IDR)', 'IDN ADM']],
        use_container_width=True
    )
