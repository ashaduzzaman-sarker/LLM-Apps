import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

## Function to return the response using the HuggingFace Model
def load_answer(question):
    # Initialize the HuggingFaceHub LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        # huggingfacehub_api_token="",
        model_kwargs={"temperature": 0}
    )

    # Create a prompt template for the question
    template = PromptTemplate(input_variables=["question"], template="{question}")
    llm_chain = LLMChain(prompt=template, llm=llm)

    # Generate the answer using the LLM chain
    answer = llm_chain.run(question)
    return answer

# App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

submit = st.button('Generate')

# If generate button is clicked
if submit and user_input:
    response = load_answer(user_input)
    st.subheader("Answer:")
    st.write(response)