import pickle
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_answer(thy_question):
    """通过RAG获得答案"""
    load_dotenv()
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url='https://twapi.openai-hk.com/v1',
        temperature=0
    )
    if st.session_state['is_new_file']:
        loader = TextLoader(f'{st.session_state["session_id"]}.txt', encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "！", "？", "，", "、", ""]
        )
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, st.session_state['em_model'])
        st.session_state['db'] = db
        st.session_state['is_new_file'] = False
    if 'db' in st.session_state:
        retriever = st.session_state['db'].as_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=st.session_state['memory']
        )
        resp = chain.invoke({'chat_history': st.session_state['memory'], 'question': thy_question})
        return resp
    return ''


st.write("## 本地知识文件智能问答工具")

if 'session_id' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )
    # UUID - Universal Unique IDentifier
    st.session_state['session_id'] = uuid.uuid4().hex
    st.session_state['is_new_file'] = False
    with open('em.pkl', 'rb') as file_obj:
        em_model = pickle.load(file_obj)
        st.session_state['em_model'] = em_model

uploaded_file = st.file_uploader('上传你的文本文件：', type="txt")
question = st.text_input('请输入你的问题', disabled=not uploaded_file)

if uploaded_file:
    st.session_state['is_new_file'] = True
    with open(f'{st.session_state["session_id"]}.txt', 'w', encoding='utf-8') as temp_file:
        temp_file.write(uploaded_file.read().decode('utf-8'))

if uploaded_file and question:
    response = get_answer(question)
    st.write('### 答案')
    st.write(response['answer'])
    st.session_state['chat_history'] = response['chat_history']

if 'chat_history' in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()