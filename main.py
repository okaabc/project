"""
数据分析智能体 - 结合优化版（修复LLM调用错误）

修复了大语言模型调用时的early_stopping_method参数错误，调整了模型配置
增加用户登录注册功能
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import logging
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, findfont
import seaborn as sns
import jieba
from wordcloud import WordCloud
import re
import os
import sys
import openpyxl
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，适合Streamlit

# 从utils导入dataframe_agent
from utils import dataframe_agent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化jieba分词
jieba.initialize()

# 中文停用词列表
STOP_WORDS = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '在', '这', '那', '有', '无', '为', '什么',
                  '如何', '怎样', '可以', '可能', '需要', '应该', '我们', '你们', '他们', '它们', '这个', '那个'])

# 设置页面配置（必须在任何Streamlit命令之前调用）
st.set_page_config(
    page_title="千锋互联数据分析智能体",
    page_icon="📊",
    layout="wide"
)

# 修复中文显示问题的全局设置
def setup_chinese_font():
    """确保中文字体正确配置"""
    try:
        # 获取系统中可用的中文字体名称
        font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'SimSun',
                      'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']

        # 查找实际可用的字体路径
        available_font_path = None
        available_font_name = None
        for font_name in font_names:
            try:
                # 尝试查找字体路径
                font_path = findfont(FontProperties(family=font_name))
                if os.path.exists(font_path):
                    available_font_path = font_path
                    available_font_name = font_name
                    break
            except:
                continue

        # 设置全局字体配置
        if available_font_path:
            # 设置 matplotlib 使用找到的字体文件
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [available_font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 设置 seaborn 使用相同字体
            sns.set(font=available_font_name)

            logger.info(f"使用中文字体: {available_font_path} (名称: {available_font_name})")
            return available_font_path, available_font_name
        else:
            # 尝试使用默认字体
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            logger.warning("未找到中文字体，使用默认字体")
            return None, None
    except Exception as e:
        logging.error(f"设置中文字体失败: {str(e)}")
        return None, None

# 初始化中文支持
font_path, font_name = setup_chinese_font()

def clean_chinese_text(text):
    """清洗中文文本数据"""
    if pd.isna(text):
        return ""
    # 移除特殊字符和标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', str(text))
    # 移除数字
    text = re.sub(r'\d+', '', text)
    # 移除空白
    return text.strip()

def create_chart(input_data, chart_type, title="数据分析结果", x_label=None, y_label=None):
    """生成统计图表（增强中文支持与多类型图表）"""
    try:
        # 检查数据有效性
        if not input_data:
            st.warning("图表数据为空")
            return False

        # 处理多种可能的输入格式
        if isinstance(input_data, dict):
            if "columns" in input_data and "data" in input_data:
                columns = input_data["columns"]
                data = input_data["data"]
            elif "x" in input_data and "y" in input_data:
                columns = input_data["x"]
                data = input_data["y"]
            elif "labels" in input_data and "values" in input_data:
                columns = input_data["labels"]
                data = input_data["values"]
            elif "data" in input_data and chart_type == "box":
                data = input_data["data"]
                columns = [f"组{i+1}" for i in range(len(data))] if isinstance(data, list) else list(data.keys())
            else:
                keys = list(input_data.keys())
                if len(keys) >= 2:
                    columns = input_data[keys[0]]
                    data = input_data[keys[1]]
                elif len(keys) == 1 and chart_type in ["hist", "box"]:
                    data = input_data[keys[0]]
                    columns = [keys[0]] * len(data) if isinstance(data, list) else list(data.keys())
                else:
                    st.warning(f"无法识别的图表数据格式: {input_data}")
                    return False
        elif isinstance(input_data, list):
            if all(isinstance(item, list) and len(item) == 2 for item in input_data):
                columns = [item[0] for item in input_data]
                data = [item[1] for item in input_data]
            elif chart_type in ["hist", "box", "pie"]:
                data = input_data
                columns = [str(i+1) for i in range(len(data))]
            else:
                st.warning(f"列表格式不适用于图表: {input_data}")
                return False
        else:
            st.warning(f"不支持的数据类型: {type(input_data)}")
            return False

        # 验证数据
        if not data or (isinstance(data, list) and len(data)) == 0:
            st.warning("缺少有效的数据")
            return False

        # 特殊处理箱线图数据
        if chart_type == "box":
            if isinstance(data, dict):
                plot_data = [d for d in data.values()]
                labels = list(data.keys())
            elif isinstance(data, list) and all(isinstance(item, list) for item in data):
                plot_data = data
                labels = columns if len(columns) == len(data) else [f"组{i+1}" for i in range(len(data))]
            else:
                plot_data = [data]
                labels = ["数据"]
        else:
            if not columns or len(columns) != len(data):
                if len(data) > 0:
                    columns = [str(i) for i in range(len(data))]
                else:
                    st.warning("缺少有效的列名或数据")
                    return False

        # 创建图表 - 确保使用中文支持
        fig, ax = plt.subplots(figsize=(8, 6))

        if chart_type == "bar":
            ax.bar(columns, data, color='steelblue')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label or "类别", fontsize=12)
            ax.set_ylabel(y_label or "数值", fontsize=12)
            plt.xticks(rotation=45, ha='right')

        elif chart_type == "line":
            ax.plot(columns, data, marker='o', linestyle='-', color='steelblue')
            y_min, y_max = min(data), max(data)
            y_range = y_max - y_min
            padding = y_range * 0.1 if y_range > 0 else 1
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
            for i, v in enumerate(data):
                ax.text(i, v + padding * 0.05, f"{v:.2f}", ha='center', va='bottom',
                        fontproperties=FontProperties(fname=font_path) if font_path else None)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label or "类别", fontsize=12)
            ax.set_ylabel(y_label or "数值", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')

        elif chart_type == "pie":
            data = [max(0, d) for d in data]
            total = sum(data)
            if total <= 0:
                st.warning("饼图数据总和必须大于0")
                return False

            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = pct * total / 100.0
                    return f'{pct:.1f}%\n({val:.1f})'
                return my_autopct

            wedges, texts, autotexts = ax.pie(
                data,
                labels=columns,
                autopct=make_autopct(data),
                startangle=90,
                wedgeprops={'edgecolor': 'white'},
                textprops={'fontsize': 10}
            )
            ax.set_title(title, fontsize=14)
            ax.axis('equal')

        elif chart_type == "hist":
            if not all(isinstance(x, (int, float)) for x in data):
                try:
                    data = [float(x) for x in data]
                except:
                    st.warning("直方图需要数值数据")
                    return False

            bins = min(20, max(5, int(len(data) ** 0.5)))
            ax.hist(data, bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label or "数值", fontsize=12)
            ax.set_ylabel(y_label or "频率", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        elif chart_type == "box":
            ax.boxplot(plot_data, labels=labels, patch_artist=True,
                       boxprops=dict(facecolor='steelblue', color='darkblue'),
                       medianprops=dict(color='yellow'))
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label or "分组", fontsize=12)
            ax.set_ylabel(y_label or "数值", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        elif chart_type == "radar":
            # 雷达图
            num_vars = len(columns)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            data += data[:1]
            angles += angles[:1]
            ax = plt.subplot(polar=True)
            ax.plot(angles, data, color='steelblue', linewidth=2)
            ax.fill(angles, data, color='steelblue', alpha=0.25)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), columns)
            ax.set_title(title, fontsize=14)

        elif chart_type == "bubble":
            # 气泡图
            if isinstance(data, dict) and 'x' in data and 'y' in data and 'size' in data:
                x_data = data['x']
                y_data = data['y']
                size_data = data['size']
            else:
                st.warning("气泡图需要x、y和size三组数据")
                return False

            if len(x_data) != len(y_data) or len(x_data) != len(size_data):
                st.warning("气泡图的三组数据长度必须相同")
                return False

            ax.scatter(x_data, y_data, s=size_data, color='steelblue', alpha=0.6)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label or "X", fontsize=12)
            ax.set_ylabel(y_label or "Y", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

        else:
            st.warning(f"不支持的图表类型: {chart_type}")
            return False

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return True

    except Exception as e:
        st.error(f"生成图表时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def display_data_table(df):
    """显示数据表格并提供下载功能"""
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="下载数据为CSV",
        data=csv,
        file_name='data_analysis.csv',
        mime='text/csv'
    )

def generate_word_cloud(text_data, title="文本关键词词云"):
    """生成中文词云图"""
    try:
        if text_data is None or (isinstance(text_data, pd.Series) and text_data.empty):
            st.warning("没有可用于生成词云的文本数据")
            return False

        if isinstance(text_data, pd.Series):
            all_text = ' '.join(text_data.astype(str).tolist())
        elif isinstance(text_data, list):
            all_text = ' '.join(str(t) for t in text_data)
        else:
            all_text = str(text_data)

        cleaned_text = clean_chinese_text(all_text)
        if not cleaned_text:
            st.warning("清洗后没有有效的文本内容")
            return False

        words = jieba.cut(cleaned_text)
        words = [word for word in words if word.strip() and word not in STOP_WORDS and len(word) > 1]
        if not words:
            st.warning("分词后没有有效关键词")
            return False

        word_freq = pd.Series(words).value_counts().to_dict()

        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
        return True

    except Exception as e:
        st.error(f"生成词云时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def data_preprocessing(df):
    """数据预处理功能"""
    st.sidebar.subheader("数据预处理选项")

    with st.sidebar.expander("处理缺失值", expanded=False):
        na_cols = df.columns[df.isna().any()].tolist()
        if na_cols:
            na_option = st.selectbox("选择处理方式", ["删除行", "填充值"], key="na_option")
            if na_option == "删除行":
                df = df.dropna()
                st.success(f"已删除包含缺失值的行，剩余 {len(df)} 行数据")
            else:
                fill_col = st.selectbox("选择填充列", na_cols, key="fill_col")
                fill_value = st.text_input("填充值", "0", key="fill_value")
                try:
                    fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
                except:
                    pass
                df[fill_col] = df[fill_col].fillna(fill_value)
                st.success(f"已填充列 {fill_col} 的缺失值")
        else:
            st.info("数据中没有缺失值")

    with st.sidebar.expander("重命名列", expanded=False):
        col_to_rename = st.selectbox("选择要重命名的列", df.columns, key="rename_col")
        new_name = st.text_input("新列名", col_to_rename, key="new_name")
        if new_name and new_name != col_to_rename:
            df = df.rename(columns={col_to_rename: new_name})
            st.success(f"已将列 '{col_to_rename}' 重命名为 '{new_name}'")

    with st.sidebar.expander("数据类型转换", expanded=False):
        col_to_convert = st.selectbox("选择要转换的列", df.columns, key="convert_col")
        current_type = str(df[col_to_convert].dtype)
        new_type = st.selectbox("选择新类型",
                                ["object", "int", "float", "datetime", "category"],
                                key="new_type")

        if st.button("应用转换", key="convert_btn"):
            try:
                if new_type == "int":
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce').astype('Int64')
                elif new_type == "float":
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce')
                elif new_type == "datetime":
                    df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors='coerce')
                elif new_type == "category":
                    df[col_to_convert] = df[col_to_convert].astype('category')
                else:  # object
                    df[col_to_convert] = df[col_to_convert].astype(str)
                st.success(f"已将列 '{col_to_convert}' 从 {current_type} 转换为 {new_type}")
            except Exception as e:
                st.error(f"转换失败: {str(e)}")

    with st.sidebar.expander("删除列", expanded=False):
        cols_to_drop = st.multiselect("选择要删除的列", df.columns, key="drop_cols")
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            st.success(f"已删除 {len(cols_to_drop)} 列")

    return df

def correlation_analysis(df):
    """数值列相关性分析"""
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("需要至少两个数值列进行相关性分析")
        return

    st.subheader("数值列相关性分析")
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("相关性热力图", fontsize=14)
    st.pyplot(fig)

    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    if high_corr:
        st.subheader("高相关性特征对 (|r| > 0.7)")
        high_corr_df = pd.DataFrame(high_corr, columns=["特征1", "特征2", "相关系数"])
        st.dataframe(high_corr_df.sort_values("相关系数", ascending=False))
    else:
        st.info("没有发现高度相关的特征对 (|r| > 0.7)")

def categorical_analysis(df, column):
    """分类变量分析"""
    st.subheader(f"分类变量分析: {column}")
    value_counts = df[column].value_counts()
    st.write(f"**类别分布统计 (共 {len(value_counts)} 个类别):**")
    st.dataframe(value_counts)

    create_chart(
        {"x": value_counts.index.astype(str).tolist(), "y": value_counts.values.tolist()},
        "bar",
        f"{column} 分布",
        x_label=column,
        y_label="数量"
    )

    sample_value = df[column].iloc[0] if len(df) > 0 else ""
    if isinstance(sample_value, str) and len(sample_value) > 10:
        st.subheader(f"{column} 文本分析")
        generate_word_cloud(df[column], f"{column} 关键词词云")

# def auto_data_analysis(df):
#     """自动数据探索分析"""
#     st.subheader("自动数据探索分析")
#     st.write("**数据概览:**")
#     st.write(f"- 行数: {len(df)}")
#     st.write(f"- 列数: {len(df.columns)}")
#     st.write(f"- 缺失值比例: {df.isna().mean().mean():.2%}")
#
#     dtype_counts = df.dtypes.value_counts()
#     st.write("**数据类型分布:**")
#     for dtype, count in dtype_counts.items():
#         st.write(f"- {dtype}: {count} 列")
#
#     numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
#     if numeric_cols:
#         st.subheader("数值列统计摘要")
#         st.dataframe(df[numeric_cols].describe())
#
#         if len(numeric_cols) > 1:
#             correlation_analysis(df)
#
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     if categorical_cols:
#         st.subheader("分类变量分析")
#         col_to_analyze = st.selectbox("选择分类列进行分析", categorical_cols)
#         categorical_analysis(df, col_to_analyze)
#
#     datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
#     if datetime_cols:
#         st.subheader("时间序列分析")
#         time_col = st.selectbox("选择时间列", datetime_cols)
#         value_col = st.selectbox("选择分析值列", numeric_cols)
#         time_df = df.set_index(time_col)[value_col].resample('M').mean()
#
#         if len(time_df) > 1:
#             create_chart(
#                 {"x": time_df.index.strftime('%Y-%m').tolist(), "y": time_df.values.tolist()},
#                 "line",
#                 f"{value_col} 月度变化趋势",
#                 x_label="时间",
#                 y_label=value_col
#             )

def auto_data_analysis(df):
    """自动数据探索分析"""
    st.subheader("自动数据探索分析")
    st.write("**数据概览:**")
    st.write(f"- 行数: {len(df)}")
    st.write(f"- 列数: {len(df.columns)}")
    st.write(f"- 缺失值比例: {df.isna().mean().mean():.2%}")

    dtype_counts = df.dtypes.value_counts()
    st.write("**数据类型分布:**")
    for dtype, count in dtype_counts.items():
        st.write(f"- {dtype}: {count} 列")

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    if numeric_cols:
        st.subheader("数值列统计摘要")
        st.dataframe(df[numeric_cols].describe())

        if len(numeric_cols) > 1:
            correlation_analysis(df)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.subheader("分类变量分析")

        # 初始化会话状态
        if 'current_category_col' not in st.session_state:
            st.session_state.current_category_col = categorical_cols[0]

        # 记录上一次选择的列
        prev_selected_col = st.session_state.current_category_col

        # 创建选择框
        col_to_analyze = st.selectbox("选择分类列进行分析", categorical_cols)

        # 当选择变化时更新会话状态并重新运行
        if col_to_analyze != prev_selected_col:
            st.session_state.current_category_col = col_to_analyze
            st.rerun()

        # 直接显示分析结果（移除expander）
        categorical_analysis(df, col_to_analyze)

    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if datetime_cols:
        st.subheader("时间序列分析")
        time_col = st.selectbox("选择时间列", datetime_cols)
        value_col = st.selectbox("选择分析值列", numeric_cols)
        time_df = df.set_index(time_col)[value_col].resample('M').mean()

        if len(time_df) > 1:
            create_chart(
                {"x": time_df.index.strftime('%Y-%m').tolist(), "y": time_df.values.tolist()},
                "line",
                f"{value_col} 月度变化趋势",
                x_label="时间",
                y_label=value_col
            )

class BasePage:
    def __init__(self, name):
        self.name = name

    def render(self):
        pass

class HomePage(BasePage):
    def render(self):
        # 检查登录状态
        if "user_name" not in st.session_state:
            st.warning("请先登录")
            return
        st.title("📊 千锋互联数据分析智能体")
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            padding: 2rem;
        }
        .stRadio > div {
            flex-direction: row;
            align-items: center;
        }
        .stRadio label {
            margin-right: 15px;
        }
        .stSidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .css-1d391kg {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .st-expander > div {
            margin-bottom: 15px;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 15px;
        }
        .st-expander > div > div:first-child {
            font-weight: bold;
            margin-bottom: 10px;
        }
        body, .stTextInput>label, .stSelectbox>label, .stNumberInput>label, 
        .stTextArea>label, .stMarkdown, .stAlert, .stButton>button {
            font-family: 'SimHei', 'Microsoft YaHei', 'STHeiti', 'SimSun', sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.sidebar.header("数据上传与处理")
            option = st.sidebar.radio("请选择数据文件类型:", ("Excel", "CSV"))
            file_type = "xlsx" if option == "Excel" else "csv"
            data = st.sidebar.file_uploader(f"上传你的{option}数据文件", type=file_type)

        if "df" not in st.session_state:
            st.session_state.df = None
        if "query_history" not in st.session_state:
            st.session_state.query_history = []

        if data:
            try:
                with st.spinner("正在加载数据..."):
                    if file_type == "xlsx":
                        sheet_names = pd.ExcelFile(data).sheet_names
                        sheet_name = st.sidebar.selectbox("选择工作表", sheet_names)
                        st.session_state.df = pd.read_excel(data, sheet_name=sheet_name)
                    else:
                        st.session_state.df = pd.read_csv(data)

                    if st.session_state.df.isnull().all().any():
                        st.sidebar.warning("检测到可能的编码问题，尝试重新加载...")
                        if file_type == "csv":
                            st.session_state.df = pd.read_csv(data, encoding='utf-8-sig')

                    st.sidebar.success("数据加载成功！")
                    st.session_state.df = data_preprocessing(st.session_state.df)

                    if st.sidebar.checkbox("执行自动数据探索分析", value=True):
                        with st.expander("原始数据预览"):
                            display_data_table(st.session_state.df.head())
                        auto_data_analysis(st.session_state.df)
            except Exception as e:
                st.error(f"读取文件时出错: {str(e)}")
                st.session_state.df = None

        with col2:
            query = st.text_area(
                "请输入你关于以上数据集的问题或数据可视化需求：",
                disabled=st.session_state.df is None,
                placeholder="例如：统计各品牌数量并绘制柱状图\n或：分析销售额与广告投入的关系",
                height=150
            )

            col_a, col_b = st.columns([1, 1])
            with col_a:
                button = st.button("生成分析报告", use_container_width=True, type="primary")
            with col_b:
                if st.button("清除历史记录", use_container_width=True):
                    st.session_state.query_history = []
                    st.rerun()

        if st.session_state.query_history:
            st.subheader("📜 历史查询记录")
            history_tab = st.tabs([f"查询 {i+1}" for i in range(len(st.session_state.query_history))])

            for i, tab in enumerate(history_tab):
                with tab:
                    q, r = st.session_state.query_history[i]
                    st.write(f"**查询内容:** {q}")

                    if "answer" in r:
                        st.markdown("**📝 分析结论:**")
                        st.write(r["answer"])

                    if "table" in r:
                        try:
                            table_data = r["table"]
                            if "columns" in table_data and "data" in table_data:
                                df_table = pd.DataFrame(
                                    table_data["data"],
                                    columns=table_data["columns"]
                                )
                                st.markdown("**📊 数据表格:**")
                                display_data_table(df_table)
                        except Exception as e:
                            st.error(f"显示表格时出错: {str(e)}")

                    chart_created = False
                    if "chart_data" in r and "chart_type" in r:
                        chart_title = r.get("title", "分析图表")
                        x_label = r.get("x_label", None)
                        y_label = r.get("y_label", None)
                        if create_chart(r["chart_data"], r["chart_type"], chart_title, x_label, y_label):
                            chart_created = True
                    elif "bar" in r:
                        if create_chart(r["bar"], "bar", "柱状图分析结果"):
                            chart_created = True
                    elif "line" in r:
                        if create_chart(r["line"], "line", "折线图分析结果"):
                            chart_created = True

                    if not chart_created and "chart_type" in r and "chart_data" in r:
                        st.warning("图表生成失败，显示原始数据")
                        st.json(r["chart_data"])

        if button and st.session_state.df is None:
            st.info("请先上传数据文件")
            st.stop()

        if button and query:
            with st.spinner("AI正在分析数据，请稍候..."):
                try:
                    result = dataframe_agent(st.session_state.df, query)
                    st.session_state.query_history.append((query, result))
                    if len(st.session_state.query_history) > 8:
                        st.session_state.query_history.pop(0)

                    st.subheader("📈 分析报告")
                    st.markdown("---")

                    if "answer" in result:
                        st.markdown("### 📝 分析结论")
                        st.write(result["answer"])

                    table_created = False
                    if "table" in result:
                        try:
                            table_data = result["table"]
                            if "columns" in table_data and "data" in table_data:
                                st.markdown("### 📊 数据表格")
                                df_table = pd.DataFrame(
                                    table_data["data"],
                                    columns=table_data["columns"]
                                )
                                display_data_table(df_table)
                                table_created = True
                        except Exception as e:
                            st.error(f"显示表格时出错: {str(e)}")

                    chart_created = False
                    chart_title = "数据分析图表"
                    x_label = result.get("x_label", None)
                    y_label = result.get("y_label", None)

                    if "chart_type" in result and "chart_data" in result:
                        chart_title = result.get("title", "数据分析图表")
                        st.markdown(f"### 📈 {chart_title}")
                        if create_chart(result["chart_data"], result["chart_type"], chart_title, x_label, y_label):
                            chart_created = True

                    if not chart_created:
                        if "bar" in result:
                            st.markdown("### 📊 柱状图分析")
                            if create_chart(result["bar"], "bar", "柱状图分析结果", x_label, y_label):
                                chart_created = True
                        elif "line" in result:
                            st.markdown("### 📈 折线图分析")
                            if create_chart(result["line"], "line", "折线图分析结果", x_label, y_label):
                                chart_created = True

                    if not any(key in result for key in ["answer", "table"]) and not chart_created:
                        st.info("未生成可视化结果，请尝试更明确的查询")
                        st.markdown("### 🧠 AI原始返回结果")
                        st.json(result)

                    st.markdown("---")
                    st.success("✅ 分析完成！您可以继续提出新的查询。")

                except Exception as e:
                    st.error(f"处理请求时出错: {str(e)}")
                    st.markdown("### 🧠 AI原始返回结果（错误时）")
                    if 'result' in locals():
                        st.json(result)

class DataOverviewPage(BasePage):
    def render(self):
        # 检查登录状态
        if "user_name" not in st.session_state:
            st.warning("请先登录")
            return
        with st.container():
            st.title('MyChatGPT')

            if 'messages' not in st.session_state:
                st.session_state['messages'] = [{'role': 'ai', 'content': '你好主人，我是你的AI助手，我叫小美'}]
                st.session_state['memory'] = ConversationBufferMemory(return_messages=True)

            for message in st.session_state['messages']:
                role, content = message['role'], message['content']
                st.chat_message(role).write(content)

            user_input = st.chat_input()
            if user_input:
                st.chat_message('human').write(user_input)
                st.session_state['messages'].append({'role': 'human', 'content': user_input})
                with st.spinner('AI正在思考，请等待……'):
                    resp_from_ai = self.get_ai_response(user_input)
                    st.session_state['history'] = resp_from_ai
                    st.chat_message('ai').write(resp_from_ai)
                    st.session_state['messages'].append({'role': 'ai', 'content': resp_from_ai})

    def get_ai_response(self, user_prompt):
        load_dotenv()
        # 修复模型名称为官方支持的gpt-3.5-turbo，移除可能导致错误的early_stopping_method参数
        model = ChatOpenAI(
            model='gpt-3.5-turbo',  # 替换为官方支持的模型名称
            base_url='https://twapi.openai-hk.com/v1',
            # 移除未定义的early_stopping_method参数
        )
        chain = ConversationChain(llm=model, memory=st.session_state['memory'])
        return chain.invoke({'input': user_prompt})['response']

class VisualizationPage(BasePage):
    def render(self):
        # 检查登录状态
        if "user_name" not in st.session_state:
            st.warning("请先登录")
            return
        with st.container():
            st.title("数据可视化")
            if "df" not in st.session_state:
                st.warning("请先在主页上传数据")
                return

            df = st.session_state["df"]

            # 检查数据集行数，提示可能的性能问题
            if len(df) > 10000:
                st.warning(f"注意：当前数据集较大（{len(df)} 行），复杂图表生成可能较慢。建议先进行数据采样。")

            # 定义图表配置映射
            CHART_CONFIG = {
                "条形图": {"type": "bar", "description": "展示分类数据的对比"},
                "折线图": {"type": "line", "description": "展示数据的趋势变化"},
                "饼图": {"type": "pie", "description": "展示数据的比例分布"},
                "直方图": {"type": "hist", "description": "展示数据的分布频率"},
                "雷达图": {"type": "radar", "description": "多维度数据的综合比较"},
                "气泡图": {"type": "bubble", "description": "展示三个变量之间的关系"}
            }

            # 数据类型分类
            numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                st.subheader("分类变量分析")

                # 初始化会话状态
                if 'current_category_col' not in st.session_state:
                    st.session_state.current_category_col = categorical_cols[0]

                # 记录上一次选择的列
                prev_selected_col = st.session_state.current_category_col

                # 创建选择框
                col_to_analyze = st.selectbox("选择分类列进行分析", categorical_cols)

                # 当选择变化时更新会话状态并重新运行
                if col_to_analyze != prev_selected_col:
                    st.session_state.current_category_col = col_to_analyze
                    st.rerun()

                # ==== 修改点：直接调用分析函数，不再使用expander包装 ====
                categorical_analysis(df, col_to_analyze)
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

            with st.expander("📊 可视化帮助", expanded=False):
                st.markdown("""
                ### 如何选择合适的图表类型？
                - **条形图**：比较不同类别的数值大小
                - **折线图**：展示数据随时间或连续变量的变化趋势
                - **饼图**：显示各部分占总体的比例关系
                - **直方图**：展示数值数据的分布情况
                - **雷达图**：多维度数据的综合比较
                - **气泡图**：展示三个变量之间的关系（X轴、Y轴和气泡大小）

                ### 数据类型建议
                - **X轴**：通常为分类变量或时间变量
                - **Y轴**：通常为数值变量
                """)

            st.subheader("自定义可视化")

            with st.form("visualization_form"):
                col1, col2 = st.columns(2)

                with col1:
                    chart_type = st.selectbox(
                        "选择图表类型",
                        list(CHART_CONFIG.keys()),
                        format_func=lambda x: f"{x} - {CHART_CONFIG[x]['description']}"
                    )

                with col2:
                    # 根据图表类型推荐合适的X轴列
                    if chart_type in ["条形图", "饼图"]:
                        recommended_x = categorical_cols
                    elif chart_type in ["折线图"]:
                        recommended_x = datetime_cols or numeric_cols
                    else:
                        recommended_x = df.columns.tolist()

                    x_column = st.selectbox(
                        "选择X轴列",
                        recommended_x,
                        key="x_column"
                    )

                # 根据图表类型调整Y轴列选择
                if chart_type == "直方图":
                    y_column = st.selectbox(
                        "选择分析列",
                        numeric_cols,
                        key="y_column_hist"
                    )
                elif chart_type == "饼图":
                    # 饼图需要聚合数据
                    aggregation = st.selectbox(
                        "选择聚合方式",
                        ["计数", "求和", "平均值"],
                        key="aggregation_pie"
                    )
                    if aggregation != "计数":
                        y_column = st.selectbox(
                            "选择数值列",
                            numeric_cols,
                            key="y_column_pie"
                        )
                elif chart_type == "气泡图":
                    y_column = st.selectbox(
                        "选择Y轴列",
                        numeric_cols,
                        key="y_column_bubble"
                    )
                    size_column = st.selectbox(
                        "选择气泡大小列",
                        numeric_cols,
                        key="size_column"
                    )
                else:
                    y_column = st.selectbox(
                        "选择Y轴列",
                        numeric_cols,
                        key="y_column_default"
                    )

                # 图表标题和标签自定义
                custom_title = st.text_input(
                    "自定义图表标题",
                    f"{y_column if 'y_column' in locals() else x_column} 分布分析"
                )

                # 高级选项
                with st.expander("高级选项", expanded=False):
                    col3, col4 = st.columns(2)
                    with col3:
                        x_label = st.text_input("X轴标签", x_column)
                    with col4:
                        y_label = st.text_input("Y轴标签", y_column if 'y_column' in locals() else "")

                    if chart_type == "饼图":
                        show_values = st.checkbox("显示具体数值", True)

                    if chart_type == "条形图" or chart_type == "折线图":
                        sort_option = st.selectbox(
                            "排序方式",
                            ["不排序", "升序", "降序"],
                            key=f"sort_{chart_type}"
                        )

                submitted = st.form_submit_button("生成图表", type="primary")

            if submitted:
                try:
                    with st.spinner("正在生成图表..."):
                        # 准备图表数据
                        chart_data = None

                        if chart_type == "饼图":
                            if aggregation == "计数":
                                plot_data = df[x_column].value_counts().reset_index()
                                plot_data.columns = [x_column, "计数"]
                                chart_data = {
                                    "labels": plot_data[x_column].tolist(),
                                    "values": plot_data["计数"].tolist()
                                }
                            elif aggregation == "求和":
                                plot_data = df.groupby(x_column)[y_column].sum().reset_index()
                                chart_data = {
                                    "labels": plot_data[x_column].tolist(),
                                    "values": plot_data[y_column].tolist()
                                }
                            elif aggregation == "平均值":
                                plot_data = df.groupby(x_column)[y_column].mean().reset_index()
                                chart_data = {
                                    "labels": plot_data[x_column].tolist(),
                                    "values": plot_data[y_column].tolist()
                                }

                        elif chart_type == "直方图":
                            chart_data = df[y_column].dropna().tolist()

                        elif chart_type == "气泡图":
                            chart_data = {
                                "x": df[x_column].tolist(),
                                "y": df[y_column].tolist(),
                                "size": df[size_column].tolist()
                            }

                        else:
                            chart_data = {
                                "x": df[x_column].tolist(),
                                "y": df[y_column].tolist()
                            }

                            # 排序处理
                            if sort_option != "不排序":
                                sorted_data = sorted(
                                    zip(chart_data["x"], chart_data["y"]),
                                    key=lambda x: x[1],
                                    reverse=(sort_option == "降序")
                                )
                                chart_data["x"], chart_data["y"] = zip(*sorted_data)

                        # 调用图表生成函数
                        create_chart(
                            chart_data,
                            CHART_CONFIG[chart_type]["type"],
                            custom_title,
                            x_label,
                            y_label
                        )

                        # 显示数据预览
                        with st.expander("查看数据预览", expanded=False):
                            if chart_type == "饼图":
                                st.dataframe(plot_data)
                            else:
                                preview_df = df[[x_column, y_column]].head(20)
                                st.dataframe(preview_df)

                except Exception as e:
                    st.error(f"生成图表时出错: {str(e)}")
                    st.warning("请检查数据类型是否与图表类型兼容，例如：直方图需要数值型数据")

class AboutPage(BasePage):
    def render(self):
        with st.container():
            st.title("关于")
            st.markdown("""
            # 四川民族学院数据分析智能体

            本应用是一个基于大语言模型的自助式数据分析工具，可以帮助用户快速分析和可视化数据。

            ## 主要功能
            - 支持Excel和CSV格式的数据文件
            - 通过自然语言查询进行数据分析
            - 自动生成统计图表和数据可视化
            - 提供数据概览和描述性统计

            ## 技术栈
            - 前端框架：Streamlit
            - 数据分析：Pandas, Matplotlib
            - 大语言模型：OpenAI GPT系列

            如有任何问题或建议，请联系我们。
            """)

def main():
    pages = {
        "主页": HomePage("主页"),
        "MyChatGPT": DataOverviewPage("MyChatGPT"),
        "数据可视化": VisualizationPage("数据可视化"),
        "关于": AboutPage("关于")
    }

    # 用户数据文件路径
    USER_DATA_FILE = "Admin.txt"

    # 确保用户数据文件存在
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            # 写入默认管理员账户
            f.write("admin,admin123\n")

    # 读取用户数据
    def read_user_data():
        users = {}
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as f:
                for line in f.readlines():
                    if line.strip():
                        username, password = line.strip().split(",")
                        users[username] = password
        return users

    # 保存用户数据
    def save_user_data(users):
        with open(USER_DATA_FILE, "w") as f:
            for username, password in users.items():
                f.write(f"{username},{password}\n")

    # 优化后的侧边栏
    with st.sidebar:
        # 添加应用标题和图标
        st.markdown("""
        <div style="display:flex; align-items:center; margin-bottom:30px;">
            <h1 style="margin:0;">📊 数据分析智能体</h1>
        </div>
        """, unsafe_allow_html=True)

        # 添加用户信息区域
        st.markdown("### 👤 用户中心")
        user_status = st.empty()

        # 显示用户状态
        if "user_name" in st.session_state:
            user_status.success(f"欢迎, {st.session_state.user_name}!")
            if st.button("退出登录"):
                del st.session_state.user_name
                st.rerun()
        else:
            with st.expander("登录/注册", expanded=True):
                login_tab, register_tab = st.tabs(["登录", "注册"])

                with login_tab:
                    username = st.text_input("用户名", key="login_username")
                    password = st.text_input("密码", type="password", key="login_password")
                    if st.button("登录", key="login_btn"):
                        users = read_user_data()
                        if username in users and users[username] == password:
                            st.session_state.user_name = username
                            st.success("登录成功!")
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")

                with register_tab:
                    new_username = st.text_input("新用户名", key="reg_username")
                    new_password = st.text_input("新密码", type="password", key="reg_password")
                    confirm_password = st.text_input("确认密码", type="password", key="reg_confirm_password")

                    if st.button("注册", key="register_btn"):
                        if new_password != confirm_password:
                            st.error("两次输入的密码不匹配")
                        elif not new_username or not new_password:
                            st.error("用户名和密码不能为空")
                        else:
                            users = read_user_data()
                            if new_username in users:
                                st.error("用户名已存在")
                            else:
                                # 保存新用户
                                users[new_username] = new_password
                                save_user_data(users)
                                st.success(f"用户 {new_username} 注册成功! 请登录")

        # 添加导航菜单
        st.markdown("---")
        st.markdown("### 🧭 导航")
        page_selection = st.radio("选择页面", list(pages.keys()), label_visibility="collapsed")

        # 检查登录状态
        is_logged_in = "user_name" in st.session_state

        # 添加数据上传区域（仅登录用户可见）
        if is_logged_in:
            st.markdown("---")
            st.markdown("### 📂 数据上传")
            option = st.radio("文件类型:", ("Excel", "CSV"), horizontal=True)
            file_type = "xlsx" if option == "Excel" else "csv"
            data = st.file_uploader(f"上传{option}文件", type=file_type, label_visibility="collapsed")

            if data:
                try:
                    with st.spinner("加载数据中..."):
                        if file_type == "xlsx":
                            sheet_names = pd.ExcelFile(data).sheet_names
                            sheet_name = st.selectbox("选择工作表", sheet_names)
                            st.session_state.df = pd.read_excel(data, sheet_name=sheet_name)
                        else:
                            st.session_state.df = pd.read_csv(data)

                        st.success("数据加载成功!")
                except Exception as e:
                    st.error(f"读取文件出错: {str(e)}")

        # 添加快速分析工具（仅登录用户可见）
        if is_logged_in and "df" in st.session_state and st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### ⚡ 快速分析工具")
            if st.button("数据概览", use_container_width=True):
                auto_data_analysis(st.session_state.df)
            if st.button("相关性分析", use_container_width=True):
                correlation_analysis(st.session_state.df)

        # 添加页脚
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#888; font-size:0.9em; margin-top:30px;">
            © 2025 四川民族学院<br>数据分析智能体
        </div>
        """, unsafe_allow_html=True)

    # 检查登录状态，未登录用户只能看到登录提示
    if "user_name" not in st.session_state:
        # 显示登录提示页面
        st.title("🔐 用户登录")
        st.markdown("""
        <style>
        .login-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 30px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-instruction {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
        }
        </style>

        <div class="login-container">
            <div class="login-header">
                <h2>数据分析智能体</h2>
                <p>请登录或注册以使用所有功能</p>
            </div>
            <div class="login-instruction">
                <p>👈 请在左侧边栏登录或注册账户</p>
                <p>👉 登录后可上传数据并进行分析</p>
                <p>🔐 您的账户信息将安全存储在本地</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 根据选择的页面渲染内容
    pages[page_selection].render()


if __name__ == "__main__":
    main()