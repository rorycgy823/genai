import streamlit as st
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Initialize the QA system
@st.cache_resource
def load_qa_system():
    """Load the QA system with caching"""
    try:
        import fixed_main
        return fixed_main.qa_chain
    except Exception as e:
        st.error(f"Failed to load QA system: {str(e)}")
        return None

# Load QA system
qa_chain = load_qa_system()

st.title("Citic智能助手")

if qa_chain is None:
    st.error("QA系统加载失败，请检查配置。")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("您的问题？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                # Use the correct method based on what's available
                if hasattr(qa_chain, 'invoke'):
                    response = qa_chain.invoke(prompt)
                elif hasattr(qa_chain, 'run'):
                    response = qa_chain.run(prompt)
                else:
                    # Fallback to calling the chain directly
                    response = qa_chain({"query": prompt})
                    if isinstance(response, dict):
                        response = response.get("result", "抱歉，无法找到相关答案。")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"处理问题时出现错误: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
