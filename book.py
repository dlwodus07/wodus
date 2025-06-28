import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정 - 와이드 레이아웃으로 설정하여 더 많은 공간 활용
st.set_page_config(
    page_title="문해력 현황 분석 및 교육 지원",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일 적용
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>📚 문해력 현황 분석 및 교육 지원 시스템</h1>
    <p>사서 및 교사를 위한 데이터 기반 문해력 교육 도구</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.title("🔧 분석 도구")
st.sidebar.markdown("---")

# 데이터 로드 함수
@st.cache_data
def load_data():
    """
    CSV 데이터를 로드하고 전처리하는 함수
    캐시를 사용하여 성능 최적화
    """
    data = {
        'Year': [2014, 2014, 2014, 2017, 2017, 2017, 2020, 2020, 2020],
        'Gender': ['전체', '남성', '여성', '전체', '남성', '여성', '전체', '남성', '여성'],
        'Value': [71.5, 77.0, 66.0, 77.6, 81.9, 73.4, 79.8, 83.7, 76.0]
    }
    df = pd.DataFrame(data)
    return df

# 데이터 분석 함수들
def calculate_gender_gap(df, year):
    """특정 연도의 성별 격차를 계산하는 함수"""
    year_data = df[df['Year'] == year]
    male_score = year_data[year_data['Gender'] == '남성']['Value'].iloc[0]
    female_score = year_data[year_data['Gender'] == '여성']['Value'].iloc[0]
    return male_score - female_score

def predict_future_literacy(df, target_year):
    """선형 회귀를 사용한 미래 문해력 예측"""
    overall_data = df[df['Gender'] == '전체'].copy()
    x = overall_data['Year'].values
    y = overall_data['Value'].values
    
    # 최소제곱법으로 기울기와 절편 계산
    slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
    intercept = np.mean(y) - slope * np.mean(x)
    
    return slope * target_year + intercept

# 데이터 로드
df = load_data()

# 사이드바 필터 옵션
st.sidebar.subheader("📊 데이터 필터")
selected_years = st.sidebar.multiselect(
    "연도 선택:",
    options=sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique())
)

selected_gender = st.sidebar.multiselect(
    "성별 선택:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

# 필터링된 데이터
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Gender'].isin(selected_gender))]

# 메인 대시보드 레이아웃
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 문해력 변화 추이")
    
    # 전체 데이터에 대한 선형 차트 (Streamlit 내장)
    overall_trend = df[df['Gender'] == '전체'].copy()
    overall_trend = overall_trend.set_index('Year')
    st.line_chart(overall_trend['Value'], height=300)
    
    # 성별별 데이터 비교를 위한 피벗 테이블 생성
    st.subheader("🔍 성별 비교 분석")
    pivot_df = filtered_df.pivot(index='Year', columns='Gender', values='Value')
    
    # 막대 차트로 성별 비교 표시
    st.bar_chart(pivot_df, height=300)
    
    # 데이터 인사이트 표시
    st.markdown("""
    <div class="insight-box">
        <h4>📊 주요 인사이트</h4>
        <ul>
            <li>2014년부터 2020년까지 전체 문해력이 꾸준히 향상되었습니다</li>
            <li>남성과 여성 간의 문해력 격차가 지속적으로 존재합니다</li>
            <li>모든 그룹에서 문해력 향상 추세를 보이고 있습니다</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 주요 통계")
    
    # 최신 데이터 (2020년) 메트릭 표시
    latest_data = df[df['Year'] == 2020]
    
    for _, row in latest_data.iterrows():
        st.metric(
            label=f"{row['Gender']} (2020년)",
            value=f"{row['Value']}%",
            delta=f"{row['Value'] - df[(df['Year'] == 2017) & (df['Gender'] == row['Gender'])]['Value'].iloc[0]:.1f}%p"
        )
    
    st.markdown("---")
    
    # 성별 격차 분석
    st.subheader("⚖️ 성별 격차 분석")
    
    # 각 연도별 성별 격차 계산 및 표시
    gap_data = []
    for year in sorted(df['Year'].unique()):
        gap = calculate_gender_gap(df, year)
        gap_data.append({'연도': year, '격차(남-여)': f"{gap:.1f}%p"})
    
    gap_df = pd.DataFrame(gap_data)
    st.dataframe(gap_df, use_container_width=True)
    
    # 격차 트렌드를 위한 라인 차트
    gap_values = [calculate_gender_gap(df, year) for year in sorted(df['Year'].unique())]
    gap_trend_df = pd.DataFrame({
        'Year': sorted(df['Year'].unique()),
        'Gap': gap_values
    }).set_index('Year')
    
    st.write("**격차 변화 추이:**")
    st.line_chart(gap_trend_df, height=200)

# 전체 너비 섹션
st.markdown("---")

# 상세 분석 섹션
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 개선율 분석")
    
    # 2014년 대비 2020년 개선율 계산
    improvement_data = []
    for gender in ['전체', '남성', '여성']:
        score_2014 = df[(df['Year'] == 2014) & (df['Gender'] == gender)]['Value'].iloc[0]
        score_2020 = df[(df['Year'] == 2020) & (df['Gender'] == gender)]['Value'].iloc[0]
        improvement = score_2020 - score_2014
        improvement_rate = (improvement / score_2014) * 100
        
        improvement_data.append({
            '성별': gender,
            '2014년': f"{score_2014}%",
            '2020년': f"{score_2020}%",
            '개선폭': f"{improvement:.1f}%p",
            '개선율': f"{improvement_rate:.1f}%"
        })
    
    improvement_df = pd.DataFrame(improvement_data)
    st.dataframe(improvement_df, use_container_width=True)

with col2:
    st.subheader("🎯 예측 분석")
    
    # 미래 예측
    future_years = [2025, 2030]
    predictions = []
    
    for year in future_years:
        predicted_value = predict_future_literacy(df, year)
        predictions.append({
            '연도': year,
            '예상 문해력': f"{predicted_value:.1f}%"
        })
    
    pred_df = pd.DataFrame(predictions)
    st.dataframe(pred_df, use_container_width=True)
    
    # 목표 설정 도구
    st.write("**목표 설정:**")
    target_year = st.selectbox("목표 연도", [2025, 2026, 2027, 2028, 2030])
    target_value = st.slider("목표 문해력 (%)", 80, 95, 85)
    
    current_value = df[df['Gender'] == '전체']['Value'].iloc[-1]
    required_improvement = target_value - current_value
    years_remaining = target_year - 2020
    annual_improvement = required_improvement / years_remaining if years_remaining > 0 else 0
    
    st.metric(
        label="연간 필요 개선율",
        value=f"{annual_improvement:.2f}%p/년"
    )

with col3:
    st.subheader("🏆 성과 지표")
    
    # 주요 성과 지표 계산
    total_improvement = df[df['Gender'] == '전체']['Value'].iloc[-1] - df[df['Gender'] == '전체']['Value'].iloc[0]
    average_annual_improvement = total_improvement / 6  # 2014~2020 = 6년
    
    # 성과 지표 표시
    performance_metrics = [
        {"지표": "전체 개선폭", "값": f"{total_improvement:.1f}%p"},
        {"지표": "연평균 개선율", "값": f"{average_annual_improvement:.2f}%p"},
        {"지표": "최고 성별 격차", "값": f"{max([calculate_gender_gap(df, year) for year in df['Year'].unique()]):.1f}%p"},
        {"지표": "최저 성별 격차", "값": f"{min([calculate_gender_gap(df, year) for year in df['Year'].unique()]):.1f}%p"}
    ]
    
    for metric in performance_metrics:
        st.markdown(f"""
        <div class="metric-card">
            <strong>{metric['지표']}</strong><br>
            <span style="font-size: 1.5em;">{metric['값']}</span>
        </div>
        """, unsafe_allow_html=True)

# 상세 데이터 테이블
st.markdown("---")
st.subheader("📋 상세 데이터 및 다운로드")

col1, col2 = st.columns([3, 1])

with col1:
    st.dataframe(filtered_df.sort_values(['Year', 'Gender']), use_container_width=True)

with col2:
    # 데이터 다운로드 버튼
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 CSV 다운로드",
        data=csv,
        file_name='literacy_data.csv',
        mime='text/csv'
    )
    
    # 보고서 생성 버튼
    report_text = f"""
문해력 분석 보고서
==================

분석 기간: 2014-2020년
데이터 포인트: {len(df)}개

주요 발견사항:
- 전체 문해력: {df[df['Gender'] == '전체']['Value'].iloc[0]}% → {df[df['Gender'] == '전체']['Value'].iloc[-1]}%
- 총 개선폭: {total_improvement:.1f}%p
- 연평균 개선율: {average_annual_improvement:.2f}%p/년

성별별 현황 (2020년):
- 남성: {df[(df['Year'] == 2020) & (df['Gender'] == '남성')]['Value'].iloc[0]}%
- 여성: {df[(df['Year'] == 2020) & (df['Gender'] == '여성')]['Value'].iloc[0]}%
- 성별 격차: {calculate_gender_gap(df, 2020):.1f}%p

권장사항:
1. 성별 격차 해소를 위한 맞춤형 프로그램 개발
2. 지속적인 문해력 향상을 위한 체계적 접근
3. 정기적인 모니터링 및 평가 시스템 구축
    """
    
    st.download_button(
        label="📄 보고서 다운로드",
        data=report_text,
        file_name='literacy_report.txt',
        mime='text/plain'
    )

# 교육 권장사항 섹션
st.markdown("---")
st.subheader("🎯 교육 권장사항 및 활용 방안")

# 탭으로 구분된 권장사항
tab1, tab2, tab3, tab4 = st.tabs(["📚 데이터 인사이트", "👨‍🏫 교사용 가이드", "📖 사서용 가이드", "📈 개선 전략"])

with tab1:
    st.markdown(f"""
    ### 🔍 데이터 분석 결과 기반 권장사항
    
    **📊 주요 발견사항:**
    - **전반적 향상**: 문해력이 지속적으로 향상되고 있음 (2014년 {df[df['Gender'] == '전체']['Value'].iloc[0]}% → 2020년 {df[df['Gender'] == '전체']['Value'].iloc[-1]}%)
    - **성별 격차 지속**: 남성이 여성보다 일관되게 높은 수준 유지
    - **2020년 격차**: 남녀 문해력 격차 {calculate_gender_gap(df, 2020):.1f}%p
    - **연평균 개선율**: 약 {average_annual_improvement:.2f}%p씩 꾸준한 향상
    
    **⚠️ 주의 필요 영역:**
    1. **성별 격차 해소**: 여성 학습자를 위한 특별 프로그램 필요
    2. **개선 속도**: 현재 속도로는 격차 해소에 시간 소요 예상
    3. **지속적 모니터링**: 정기적인 평가와 조정 필요
    
    **🎯 2025년 예상 문해력**: {predict_future_literacy(df, 2025):.1f}%
    """)

with tab2:
    st.markdown("""
    ### 👨‍🏫 교사를 위한 실전 가이드
    
    **📋 수업 계획 수립:**
    - **차별화 교육**: 성별별 학습 특성을 고려한 맞춤형 접근
    - **조기 개입**: 문해력 부족 학생 조기 발견 시스템 구축
    - **협력 학습**: 동료 교수법(Peer Teaching) 활용으로 상호 학습 촉진
    - **다양한 텍스트**: 장르별, 난이도별 다양한 읽기 자료 제공
    
    **📊 평가 및 피드백:**
    - **정기 진단**: 학기별 문해력 수준 평가 실시
    - **개별 추적**: 학생별 진도 모니터링 시스템 구축
    - **즉시 피드백**: 읽기 활동 중 실시간 지도 및 교정
    - **포트폴리오**: 학생의 문해력 성장 과정 체계적 기록
    
    **🎯 성별 격차 해소 전략:**
    - 여학생 친화적 읽기 자료 확충
    - 소그룹 토론 활동을 통한 참여 증대
    - 멘토링 프로그램 운영
    """)

with tab3:
    st.markdown("""
    ### 📖 사서를 위한 전문 가이드
    
    **📚 장서 개발 전략:**
    - **수준별 분류**: 문해력 단계별 도서 체계적 분류
    - **성별 고려**: 남녀 선호도를 반영한 균형 잡힌 장서 구성
    - **다양한 매체**: 전자책, 오디오북 등 다양한 형태 자료 확충
    - **지속적 갱신**: 최신 트렌드를 반영한 신간 도서 적극 수집
    
    **🎯 맞춤형 프로그램 운영:**
    - **독서 동아리**: 성별, 연령별 특성을 고려한 동아리 운영
    - **원북원시티**: 지역사회 전체가 함께하는 독서 운동 전개
    - **독서 치료**: 문해력 부족 이용자를 위한 전문 프로그램
    - **디지털 리터러시**: 21세기 필수 소양인 디지털 문해력 교육
    
    **📊 이용자 맞춤 서비스:**
    - 개인별 도서 추천 시스템 구축
    - 독서 진도 관리 및 상담 서비스
    - 문해력 향상 워크숍 정기 개최
    """)

with tab4:
    st.markdown(f"""
    ### 📈 체계적 문해력 개선 전략
    
    **🚀 단기 전략 (1년 이내):**
    - **긴급 지원**: 성별 격차 해소를 위한 특별 프로그램 즉시 시행
    - **역량 강화**: 교사 및 사서 대상 문해력 교육 전문 연수 확대
    - **환경 조성**: 가정-학교-도서관 연계 독서 환경 구축
    - **평가 시스템**: 정기적 문해력 진단 도구 개발 및 적용
    
    **🎯 중기 전략 (2-3년):**
    - **커리큘럼 개발**: 체계적인 문해력 교육과정 설계 및 시행
    - **인프라 구축**: 디지털 문해력 교육을 위한 기술 인프라 확충
    - **네트워크 강화**: 교육기관 간 협력 체계 구축 및 운영
    - **성과 관리**: 데이터 기반 성과 측정 및 환류 시스템 정착
    
    **🌟 장기 전략 (5년 이상):**
    - **사회적 문해력**: 지역사회 전체의 문해력 향상 생태계 조성
    - **국제 수준**: 글로벌 기준에 부합하는 문해력 수준 달성
    - **지속가능성**: 자율적이고 지속가능한 문해력 향상 문화 정착
    
    **📊 목표 수치:**
    - 2025년까지 전체 문해력 85% 달성
    - 성별 격차 5%p 이하로 단축
    - 연간 개선율 2%p 이상 유지
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>
    <h3>📚 문해력 현황 분석 및 교육 지원 시스템</h3>
    <p><strong>사서 및 교사를 위한 데이터 기반 교육 도구</strong></p>
    <p>💡 데이터 기반으로 더 나은 교육 환경을 만들어갑니다</p>
    <p>🎯 모든 학습자의 문해력 향상을 위해 함께 노력합시다</p>
</div>
""", unsafe_allow_html=True)
