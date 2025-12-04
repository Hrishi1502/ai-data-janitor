import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. CONFIGURATION
st.set_page_config(page_title="Universal Data Janitor", page_icon="🧹", layout="centered")

# 2. API KEY CHECK
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("⚠️ System Error: API Key not working. Please contact the administrator.")
    st.stop()

# 3. HELPER: DEEP SCAN
def analyze_data(df):
    summary = []
    summary.append(f"Total Rows: {len(df)}")
    summary.append(f"Total Columns: {len(df.columns)}")
    
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        # Take a sample but convert to list to avoid dataframe slicing issues
        unique_sample = list(df[col].unique()[:5]) 
        col_info = f"- Column '{col}': Type={dtype}, Missing={nulls}, Sample Values={unique_sample}"
        summary.append(col_info)
        
    return "\n".join(summary)

# 4. HELPER: GENERATE SUGGESTION
def generate_cleaning_suggestion(df, llm):
    data_profile = analyze_data(df)
    
    audit_prompt = f"""
    You are a Data Cleaning Expert. Analyze this dataset profile.
    
    DATASET FACTS:
    {data_profile}
    
    YOUR TASK:
    Write a concise cleaning plan.
    
    RULES:
    1. If a column has missing values, suggest how to fill them.
    2. If a column looks like a categorical code (e.g., 0/1), suggest mapping it.
    3. If there are duplicates, suggest dropping them.
    
    OUTPUT:
    Just the instructions. Example: "Fill missing Age with median. Drop duplicates."
    """
    
    response = llm.invoke(audit_prompt)
    return response.content

# 5. UI SETUP
st.title("🧹 Universal Data Janitor")
st.markdown("### Intelligent Auto-Cleaning")

# 6. MAIN APP
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    # --- ROW COUNT DISPLAY (The Trust Fix) ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Row Count", len(df))
    with col2:
        st.metric("Column Count", len(df.columns))

    st.write("#### 🔍 Data Preview (First 5 Rows Only)")
    st.dataframe(df.head(), use_container_width=True)
    
    # AUTO-GENERATE SUGGESTION
    if st.session_state.suggestion == "":
        with st.spinner("🤖 Scanning full dataset..."):
            suggestion = generate_cleaning_suggestion(df, llm)
            st.session_state.suggestion = suggestion

    st.write("#### 📝 Cleaning Plan")
    user_query = st.text_area(
        "Instructions",
        value=st.session_state.suggestion,
        height=100,
        label_visibility="collapsed"
    )

    run_btn = st.button("🚀 Run Cleaning", type="primary")

    if run_btn:
        with st.spinner("Agent is processing ALL rows..."):
            try:
                df_to_clean = df.copy()
                
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df_to_clean, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    agent_type="zero-shot-react-description",
                    handle_parsing_errors=True
                )
                
                # --- THE ANTI-TRUNCATION PROMPT ---
                final_prompt = f"""
                You are a Python Data Analyst working with a dataframe named `df`.
                
                USER INSTRUCTIONS: 
                "{user_query}"
                
                CRITICAL EXECUTION RULES:
                1. Modify `df` IN PLACE. 
                2. DO NOT resize or truncate the dataframe unless explicitly asked to "drop rows".
                3. NEVER run `df = df.head()` or `df = df[:5]`. processing must happen on the ENTIRE dataset.
                4. If the user asks to drop duplicates, use `df.drop_duplicates(inplace=True)`.
                5. Output "Final Answer: Done" when finished.
                """
                
                result = agent.invoke(final_prompt)
                
                st.success("✅ Cleaning Complete!")
                
                # --- VERIFICATION METRICS ---
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", len(df))
                with col2:
                    # Highlight if rows were dropped
                    delta = len(df_to_clean) - len(df)
                    st.metric("Final Rows", len(df_to_clean), delta=delta)
                
                st.dataframe(df_to_clean.head(), use_container_width=True)
                
                csv = df_to_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Clean CSV",
                    data=csv,
                    file_name="cleaned_data_full.csv",
                    mime="text/csv",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"❌ Execution Error: {e}")