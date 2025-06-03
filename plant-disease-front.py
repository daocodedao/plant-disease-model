import streamlit as st
import requests
from PIL import Image
import io
import base64
import json

# -------------------------------------
# 📌 Streamlit 页面配置
# -------------------------------------
st.set_page_config(
    page_title="植物病害识别系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------
# 📌 Custom CSS for UI improvement
# -------------------------------------
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 8px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #45a049; }
    .sidebar .sidebar-content { background-color: #f1f8e9; }
    h1, h2, h3 { color: #2e7d32; }
    .report-container { background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
    .healthy-tag { background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 4px; font-size: 14px; }
    .disease-tag { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 4px; font-size: 14px; }
    .info-box { background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------
# 📌 Backend API Configuration
# -------------------------------------
import sys
# API_ENDPOINT = "http://39.105.194.16:8503"  # Change this to match your backend API endpoint
if sys.platform == 'darwin':
    API_ENDPOINT = "http://0.0.0.0:8503"
else:
    API_ENDPOINT = "http://39.105.194.16:8503" 


def analyze_image(image_file):
    """Send image to backend API for analysis"""
    try:
        # Convert image to base64
        img_bytes = image_file.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        # Send to backend
        reUrl = f"{API_ENDPOINT}/predict"
        response = requests.post(
            reUrl,
            json={"image": encoded, "filename": image_file.name}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Connection Error: {reUrl} {str(e)}"}

# -------------------------------------
# 📌 Initialize Session State
# -------------------------------------
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

# -------------------------------------
# 📌 Sidebar Navigation
# -------------------------------------
with st.sidebar:
    st.title("植物护理")
    st.markdown("### 🌿 AI植物医生")
    
    st.markdown("---")
    app_mode = st.radio("导航", ["首页", "病害识别", "植物护理指南", "关于我们"])
    
    st.markdown("---")
    st.markdown("#### 使用说明")
    st.info("""
    1. 上传一张清晰的植物叶片图片
    2. 点击'分析图片'
    3. 查看诊断结果和治疗方案
    """)
    
    st.markdown("---")
    st.markdown("#### 由植物护理团队用❤️开发")

# -------------------------------------
# 📌 Main Page Logic
# -------------------------------------
if app_mode == "首页":
    st.title("🌿 植物病害识别系统")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("images/homepage.jpg", use_container_width=True)
    
    with col2:
        st.markdown("""
        ## 欢迎使用植物护理！👋
        
        我们的AI驱动工具可以帮助您：
        
        - **识别**植物病害，快速准确
        - **了解**病害原因和症状
        - **获取**个性化治疗建议
        - **预防**未来的植物健康问题
        
        只需上传植物叶片的照片，我们的AI就能完成剩下的工作！
        """)
    
    st.markdown("---")
    
    st.header("常见植物病害")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("番茄晚疫病")
        st.image("images/Tomato_Late_blight.jpg", use_container_width=True)
        st.markdown("一种严重的病害，导致褐色病斑和白色霉状物。")
        
    with col2:
        st.subheader("苹果黑星病")
        st.image("images/apple-scab.jpg", use_container_width=True)
        st.markdown("在叶片和果实上造成深色、scar痕状病斑。")
        
    with col3:
        st.subheader("叶斑病")
        st.image("images/leaf-spot.jpg", use_container_width=True)
        st.markdown("常见的真菌感染，表现为圆形斑点和叶片发黄。")

elif app_mode == "病害识别":
    st.title("🔍 病害识别")
    
    st.markdown("""
    上传一张清晰的植物叶片图片来诊断潜在的病害。
    为获得最佳结果，请确保光线充足并聚焦于受影响区域。
    """)
    
    uploaded_file = st.file_uploader("上传图片...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(uploaded_file, caption="已上传图片", use_container_width=True)
        
        with col2:
            analyze_clicked = st.button("🔬 分析图片", key="analyze_button")
            
            if analyze_clicked:
                with st.spinner('正在分析您的植物...'):
                    result = analyze_image(uploaded_file)
                    
                    if "error" in result:
                        st.error(f"错误: {result['error']}")
                    else:
                        st.session_state['analyzed'] = True
                        st.session_state['prediction_result'] = result

        if st.session_state['analyzed'] and st.session_state['prediction_result']:
            result = st.session_state['prediction_result']
            
            st.markdown("---")
            st.header("诊断结果")
            
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            
            if "healthy" in result["disease_name"].lower():
                st.markdown(f'<span class="healthy-tag">健康</span>', unsafe_allow_html=True)
                st.success(f"好消息！您的植物看起来很健康。")
            else:
                st.markdown(f'<span class="disease-tag">发现病害</span>', unsafe_allow_html=True)
                st.warning(f"检测到病害: {result['disease_name'].replace('___', ' - ').replace('_', ' ')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if "healthy" not in result["disease_name"].lower():
                st.markdown("---")
                st.header("病害信息")
                
                tabs = st.tabs(["描述", "症状", "治疗", "预防"])
                
                with tabs[0]:
                    st.markdown(result.get("description", "暂无描述信息"))
                
                with tabs[1]:
                    st.markdown(result.get("symptoms", "暂无症状信息"))
                
                with tabs[2]:
                    st.markdown(result.get("treatment", "暂无治疗信息"))
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("⚠️ **提醒**: 在应用任何治疗方案前，请务必咨询专业人士。")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[3]:
                    st.markdown(result.get("prevention", "暂无预防信息"))
                
                if "videos" in result:
                    st.markdown("---")
                    st.subheader("相关资源")
                    st.markdown(result["videos"])

elif app_mode == "植物护理指南":
    st.title("🌱 植物护理指南")
    
    st.markdown("""
    探索我们全面的植物健康护理指南。
    """)
    
    care_topics = [
        "浇水基础", 
        "营养管理", 
        "病虫害预防", 
        "季节性护理",
        "室内植物技巧"
    ]
    
    selected_topic = st.selectbox("选择主题:", care_topics)
    
    if selected_topic == "浇水基础":
        st.subheader("浇水基础")
        st.markdown("""
        ### 关键原则
        - 深浇水但频率要低，以促进根系深层生长
        - 在植物根部浇水，避免弄湿叶片
        - 最佳浇水时间是早晨
        
        ### 浇水过多的迹象
        - 叶片发黄
        - 茎秆变软发烂
        - 土壤表面出现霉菌
        
        ### 浇水不足的迹象
        - 土壤湿润但植物萎蔫
        - 叶片边缘发脆变褐
        - 生长缓慢
        """)
    
    elif selected_topic == "营养管理":
        st.subheader("营养管理")
        st.markdown("""
        ### 基本营养元素
        - **氮(N)**: 促进叶片生长和绿色素形成
        - **磷(P)**: 促进根系生长、开花和结果
        - **钾(K)**: 增强整体健康和抗病能力
        
        ### 有机肥料vs化肥
        有机肥料缓慢释放养分并改善土壤结构。
        化肥提供即时养分但不改善土壤。
        
        ### 施肥技巧
        - 严格按照包装说明使用 - 过量施肥反而有害！
        - 在湿润的土壤中施肥，防止根系灼伤
        - 秋冬季生长缓慢时减少施肥
        """)

elif app_mode == "关于我们":
    st.title("关于植物护理")
    
    st.markdown("""
    ## 我们的使命
    
    在植物护理，我们相信每个人都应该拥有健康茁壮的植物。我们的AI驱动工具让专业的植物病害诊断
    服务变得触手可及，无论是业余园艺爱好者还是专业农民都能受益。
    
    ## 技术实力
    
    我们的系统使用深度学习卷积神经网络(CNN)，在超过87,000张植物叶片图片上进行训练，
    涵盖38种不同的植物病害类别和健康植物。
    
    ## 主要特点
    
    - **快速分析**: 几秒钟内获得结果
    - **详细信息**: 了解病因、症状和治疗方法
    - **预防建议**: 避免未来发生病害
    - **高准确度**: 我们的模型在测试数据集上达到96%以上的准确率
    
    ## 开发团队
    
    我们的跨学科团队汇集了以下领域的专业知识：
    - 机器学习与人工智能
    - 植物病理学
    - 农业科学
    - 软件开发
    """)
    
    st.markdown("---")
    
    st.subheader("隐私与数据使用")
    st.info("""
    您上传的图片会被安全处理，除非您明确同意用于改进模型，
    否则不会被永久存储。
    """)
