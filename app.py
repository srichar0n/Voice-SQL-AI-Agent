import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

openai_key = st.secrets["OPENAI_API_KEY"]
#os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Path to your SQLite database file
database_file_path = "./db/salary.db"

# Create an engine to connect to the SQLite database
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

MSSQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.
"""

db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

sql_agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    top_k=30,
    verbose=True,
)

# Streamlit interface with styling
st.set_page_config(
    page_title="SQL Query AI Agent",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Center the title */
    .css-10trblm {text-align: center;}
    .css-1d391kg {text-align: center;}
    /* Add background color */
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    /* Style buttons */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    /* Style headers */
    h1 {
        color: #333333;
        font-size: 2.5em;
    }
    h2 {
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("Voice-SQL AI Agent")
st.markdown(
    """
    Welcome to the **SQL Query AI Agent**! This app allows you to:
    - Query a database using natural language.
    - Use voice input for queries.
    """
)

# Function to capture voice input and convert to text
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your query... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing your voice input...")
            query = recognizer.recognize_google(audio)
            st.success(f"Recognized query: {query}")
            return query.strip()  # Strip extra spaces
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return None

# Initialize session state for the question
if "question" not in st.session_state:
    st.session_state["question"] = ""

# Option to choose between text or voice input
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ("Text", "Voice"))

if input_mode == "Text":
    st.session_state["question"] = st.text_input("Enter your query:", value=st.session_state["question"])
elif input_mode == "Voice":
    if st.button("Record Voice Query"):
        voice_query = get_voice_input()
        if voice_query:
            st.session_state["question"] = voice_query

# Display the current query for debugging
st.write(f"Current query: {st.session_state['question']}")

if st.button("Run Query"):
    if st.session_state["question"]:
        try:
            # Debugging: Log the query
            st.info(f"Processing query: {st.session_state['question']}")

            # Invoke the SQL agent
            res = sql_agent.invoke(st.session_state["question"])

            # Debugging: Log the raw output
            st.info("Query executed successfully. Raw output:")
           # st.json(res)  # Display the raw response for debugging

            # Display the final output
            if "output" in res:
                st.markdown(res["output"])
            else:
                st.error("No output returned by the SQL agent.")
        except Exception as e:
            # Handle errors gracefully
            st.error(f"An error occurred while processing the query: {e}")
    else:
        st.error("Please provide a query.")