"""最小化测试 — 排除 Streamlit 本身问题."""
import streamlit as st

st.set_page_config(page_title="测试", page_icon="X", layout="wide")

st.title("最小化测试")
st.write("如果你能看到这句话，说明 Streamlit 本身没问题")

if "count" not in st.session_state:
    st.session_state.count = 0

st.write(f"计数器: {st.session_state.count}")

if st.button("+1"):
    st.session_state.count += 1
    st.rerun()
