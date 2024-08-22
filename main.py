import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

from graph.graph import app

st.title("NumPy Fundamentals Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    def response_generator(prompt):
        try:
            # Call the LLM model to generate a response
            llm_response = app.invoke(input={"question": prompt})
            response_text = llm_response.get("generation", "")
            for word in response_text.split():
                yield word + " "
                time.sleep(0.05)
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        response = "".join(response_generator(prompt))
        #print(response)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


