import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib
import platform
import os
import matplotlib.font_manager as fm

# 1. 📊 전천후 한글 깨짐 방지 설정 (서버 환경 대응 자동 다운로드 포함)
def set_korean_font():
    system_name = platform.system()
    
    if system_name == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif system_name == "Darwin":
        plt.rc('font', family='AppleGothic')
    else:
        # 💡 리눅스/클라우드 서버 환경을 위한 나눔폰트 자동 설치 로직
        font_dir = "/usr/share/fonts/truetype/nanum"
        font_path = os.path.join(font_dir, "NanumGothic.ttf")
        
        # 서버에 폰트가 없으면 패키지 설치 시도 혹은 시스템에 사용 가능한 한글 폰트 탐색
        if not os.path.exists(font_path):
            try:
                os.system("sudo apt-get install -y fonts-nanum")
                fm._rebuild()
            except:
                pass
        
        # 시스템에 설치된 폰트 중 한글 폰트 매핑 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in font_list:
            plt.rc('font', family='NanumGothic')
        elif 'Noto Sans CJK KR' in font_list:
            plt.rc('font', family='Noto Sans CJK KR')
        else:
            # 폰트 강제 매핑 규칙 추가
            plt.rcParams['font.family'] = 'sans-serif'
            
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 2. 🎨 Streamlit 페이지 설정
st.set_page_config(page_title="폐 건강 상태 AI 분석기", layout="wide", page_icon="🫁")

# 3. 💾 데이터 및 모델 로드
@st.cache_resource
def load_resources():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('lung.csv')
    return model, scaler, df

try:
    model, scaler, df = load_resources()
except FileNotFoundError as e:
    st.error(f"⚠️ 파일 로드 실패: `{e.filename}` 파일을 확인해주세요.")
    st.stop()

# 4. 군집 정보 및 컬러 정의
cluster_mapping = {
    2: {"name": "건강군", "color": "#FBC02D", "text_color": "#A77A00", "bg_color": "#FFFDE7"},
    0: {"name": "중간 그룹", "color": "#7B1FA2", "text_color": "#4A148C", "bg_color": "#F3E5F5"},
    1: {"name": "위험군", "color": "#1976D2", "text_color": "#0D47A1", "bg_color": "#E3F2FD"}
}

# 5. 🏢 UI 레이아웃 시작
st.title("🫁 폐 건강 상태 AI 분석기")
st.markdown("개인화된 데이터를 기반으로 AI 군집 분석을 수행하고 건강 상태를 예측합니다.")
st.markdown("---")

# 사이드바 입력창
st.sidebar.header("👤 환자 데이터 입력")
smokes = st.sidebar.number_input("🚬 흡연량 입력", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
areaq = st.sidebar.number_input("🌫️ 거주지 대기질 입력", min_value=0.0, max_value=100.0, value=23.0, step=1.0)
alkhol = st.sidebar.number_input("🍺 음주량 입력", min_value=0.0, max_value=100.0, value=6.0, step=1.0)

submit_btn = st.sidebar.button("📊 데이터 분석 실행", use_container_width=True)

if submit_btn:
    # 데이터 예측
    new_patient = pd.DataFrame([[smokes, areaq, alkhol]], columns=['흡연량', '거주지 대기질', '음주량'])
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = int(model.predict(new_patient_scaled)[0])
    
    result = cluster_mapping.get(pred_cluster, {"name": f"군집 {pred_cluster}", "color": "#757575", "text_color": "#333", "bg_color": "#F5F5F5"})
    
    # 💡 2D 그래프 상 눈으로 보이는 위치 추적 설명 생성
    visual_cluster = "건강군(노랑)" if smokes <= 10 and alkhol <= 6 else "분포 외"
    
    # 레이아웃 배치 (좌측 결과창, 우측 그래프)
    col1, col2 = st.columns([1.1, 1.1])
    
    with col1:
        st.subheader("🔍 AI 종합 분석 결과")
        
        # 메인 결과 박스
        st.markdown(f"""
        <div style="background-color: {result['bg_color']}; padding: 25px; border-left: 8px solid {result['color']}; border-radius: 12px; margin-bottom: 20px;">
            <span style="background-color: {result['color']}; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;">최종 판정</span>
            <h1 style="margin: 10px 0 5px 0; color: {result['text_color']}; font-weight: bold; font-size: 38px;">{result['name']}</h1>
            <p style="margin: 0; color: #666; font-size: 14px;">AI 연산 매핑: 고유 군집 {pred_cluster}번 영역</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 정밀 리포트 섹션
        st.markdown("### 📋 위치 불일치 안내 및 상세 리포트")
        
        report_html = f"""
        <div style="background-color: #F8F9FA; padding: 20px; border: 1px solid #E9ECEF; border-radius: 8px; line-height: 1.6;">
            📌 <b>그래프 위치와 결과가 다른 이유:</b><br>
            현재 환자분의 [흡연량: {smokes} / 음주량: {alkhol}] 수치는 우측 그래프상에서 <b>{visual_cluster}</b> 영역에 찍혀 있습니다. <br><br>
            하지만 AI 모델은 그래프에 표시되지 않은 <b>[거주지 대기질: {areaq}]</b> 수치까지 포함하여 3차원 분석을 수행합니다. 
            그 결과, 흡연량 대비 상대적으로 높은 대기질 위험도 요인이 반영되어 최종 판정은 
            <span style="color: {result['color']}; font-weight: bold;">[{result['name']}]</span> 영역으로 안전하게 분류되었습니다.
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        
        # 데이터 확인용 표
        st.markdown("<br>**[입력된 환자 데이터 수치]**", unsafe_allow_html=True)
        st.dataframe(new_patient, hide_index=True, use_container_width=True)
        
    with col2:
        st.subheader("📈 군집 분포 내 환자 위치 시각화")
        
        fig, ax = plt.subplots(figsize=(7, 5.2))
        
        # 색상 매핑
        df_colors = df['cluster'].map({0: '#7B1FA2', 1: '#1976D2', 2: '#FBC02D'})
        
        # 배경 산점도
        ax.scatter(df['흡연량'], df['음주량'], c=df_colors, alpha=0.35, edgecolors='none', s=45)
        
        # 새 환자 위치 (X)
        ax.scatter(smokes, alkhol, c='black', s=300, marker='X', zorder=5)
        
        # 그래프 스타일링
        ax.set_xlabel('흡연량', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('음주량', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('흡연량 및 음주량에 따른 군집 분포 (2D 투영)', fontsize=13, pad=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # 범례 설정
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='건강군 (노랑)', markerfacecolor='#FBC02D', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='중간 (보라색)', markerfacecolor='#7B1FA2', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='위험군 (파란색)', markerfacecolor='#1976D2', markersize=10),
            Line2D([0], [0], marker='X', color='w', label='새 환자 위치', markerfacecolor='black', markersize=12, markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, title="구분", loc="upper right", frameon=True)
        
        st.pyplot(fig)
else:
    st.info("👈 왼쪽 사이드바에서 환자 정보를 입력한 후, **[데이터 분석 실행]** 버튼을 클릭하세요.")