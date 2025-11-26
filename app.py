import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import warnings

# 경고 메시지 무시 설정 (Streamlit 환경에서 발생하는 불필요한 경고를 숨깁니다)
warnings.filterwarnings("ignore")

# RGB to HEX 변환 함수
def rgb_to_hex(rgb):
    # RGB 값이 0-255 범위를 벗어날 경우를 대비하여 클램프(Clamp) 처리
    rgb = np.clip(rgb, 0, 255).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

# 핵심 색상 분석 및 팔레트 생성 함수
def analyze_image_colors(uploaded_file, num_colors=5):
    """업로드된 파일에서 대표 색상을 추출하고 시각화합니다."""

    # PIL(Pillow) 라이브러리를 사용하여 이미지 열기
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    # OpenCV 처리를 위해 numpy 배열로 변환
    img = np.array(img_pil)

    # 이미지를 2차원 배열로 평탄화: (픽셀 수, 3) 형태로 변환
    pixels = img.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # K-Means 클러스터링
    # st.spinner를 사용하여 사용자에게 처리 중임을 알립니다.
    with st.spinner(f"이미지에서 {num_colors}개 대표 색상을 추출 중입니다..."):
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
        kmeans.fit(pixels)

    # 클러스터 중심(대표 색상) 추출
    colors = kmeans.cluster_centers_.astype(int)
    
    # 각 클러스터에 속한 픽셀의 개수와 색상 비율 계산
    label_counts = np.bincount(kmeans.labels_)
    combined = sorted(zip(label_counts, colors), key=lambda x: x[0], reverse=True)
    sorted_colors = [item[1] for item in combined]
    sorted_counts = [item[0] for item in combined]
    
    total_pixels = sum(sorted_counts)
    proportions = [count / total_pixels for count in sorted_counts]

    hex_colors = [rgb_to_hex(color) for color in sorted_colors]

    # --- 시각화 결과를 Streamlit에 출력 ---
    st.subheader("📊 분석 결과")
    
    # 1. 원본 이미지 출력
    st.image(img_pil, caption='업로드된 원본 이미지', use_column_width=True)
    
    # 2. 색상 팔레트 시각화 (Matplotlib 사용)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 원형 차트 생성
    wedges, texts, autotexts = ax.pie(
        proportions, 
        labels=hex_colors, 
        colors=[c / 255.0 for c in sorted_colors], 
        autopct='%1.1f%%', 
        startangle=90, 
        textprops={'color':"black", 'fontsize':10}
    )
    ax.axis('equal') # 원형을 유지
    ax.set_title(f"추출된 대표 색상 팔레트 ({num_colors}개)", fontsize=14)
    
    st.pyplot(fig) # Streamlit에 Matplotlib 그림을 표시

    # 3. HEX 코드 표 출력
    st.markdown("### 📋 추출된 색상 팔레트 (HEX 코드)")
    
    data = {'색상 순위': [f"색상 {i+1}" for i in range(num_colors)], 
            'HEX 코드': hex_colors, 
            '비율': [f"{p*100:.1f} %" for p in proportions]}
            
    st.table(data)
    
    st.success("✅ 색상 분석이 완료되었습니다. 디자인에 바로 활용해 보세요!")

# =================================================================
#                         Streamlit UI 구성
# =================================================================

st.set_page_config(page_title="🎨 이미지 색상 팔레트 생성기", layout="wide")

st.title("🎨 이미지 기반 색상 팔레트 생성기")
st.markdown("---")
st.write("이미지를 업로드하면 K-Means 클러스터링 알고리즘을 사용하여 이미지의 **대표 색상 팔레트**를 추출해 드립니다. 광고 디자인의 톤앤매너 설정에 활용하세요.")

# 사이드바 설정
st.sidebar.header("설정")
num_colors_select = st.sidebar.slider("추출할 색상 개수 (K)", 2, 10, 5)

# 파일 업로더
uploaded_file = st.file_uploader("🖼️ 분석할 이미지를 선택하세요.", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 파일이 업로드되면 분석 함수 실행
    analyze_image_colors(uploaded_file, num_colors_select)
else:
    st.info("⬆️ 분석을 시작하려면 이미지를 업로드하세요.")
