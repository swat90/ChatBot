import streamlit as st
import os
import pandas as pd
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_mic_recorder import speech_to_text
from model_pipelineV2 import ModelPipeLine
from q_learning_chatbot import QLearningChatbot

from gtts import gTTS
from io import BytesIO
st.set_page_config(page_title="PeacePal") 
#image to the sidebar
#image_path = os.path.join('images', 'sidebar.jpg')
#st.sidebar.image(image_path, use_column_width=True)

st.title('PeacePal 🌱')

mdl = ModelPipeLine()
# Now you can access the retriever attribute of the ModelPipeLine instance
# retriever = mdl.retriever

final_chain = mdl.create_final_chain()

# Define states and actions
states = [
    "Negative",
    "Neutral",
    "Positive",
]

# Initialize Q-learning chatbot and mental health classifier
chatbot = QLearningChatbot(states)

# Function to display Q-table
def display_q_table(states):
    values = [0,1,2]
    q_table_dict = {"State": states,
                    "values":values}
    q_table_df = pd.DataFrame(q_table_dict)
    return q_table_df

def text_to_speech(text):
    # Use gTTS to convert text to speech
    tts = gTTS(text=text, lang="en")
    # Save the speech as bytes in memory
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp


def speech_recognition_callback():
    # Ensure that speech output is available
    if st.session_state.my_stt_output is None:
        st.session_state.p01_error_message = "Please record your response again."
        return

    # Clear any previous error messages
    st.session_state.p01_error_message = None

    # Store the speech output in the session state
    st.session_state.speech_input = st.session_state.my_stt_output 

## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm your Mental health Assistant, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi']
    
if "entered_text" not in st.session_state:
    st.session_state.entered_text = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_sentiment" not in st.session_state:
    st.session_state.user_sentiment = "Neutral"

# Layout of input/response containers

colored_header(label='', description='', color_name='blue-30')
response_container = st.container()
input_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def generate_response(prompt):
    sentiment = mdl.predict_classification(prompt)
    response = mdl.call_conversational_rag(prompt,final_chain)
    return response['answer'], sentiment

# Collect user input
# Add a radio button to choose input mode
input_mode = st.sidebar.radio("Select input mode:", ["Text", "Speech"])
user_message = None
if input_mode == "Speech":
    # Use the speech_to_text function to capture speech input
    speech_input = speech_to_text(key="my_stt", callback=speech_recognition_callback)
    # Check if speech input is available
    if "speech_input" in st.session_state and st.session_state.speech_input:
        # Display the speech input
        # st.text(f"Speech Input: {st.session_state.speech_input}")

        # Process the speech input as a query
        user_message = st.session_state.speech_input
        st.session_state.speech_input = None
else:
    user_message = st.chat_input("Type your message here:")

## Applying the user input box        
with input_container:
    if user_message:
        detected_sentiment = None
        st.session_state.entered_text.append(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Display the user's message
        with st.chat_message("user"):
            st.write(user_message)
            
        # Process the user's message and generate a response
        with st.spinner("Processing..."):
            response,detected_sentiment = generate_response(user_message)
            st.session_state.past.append(user_message)
            st.session_state.messages.append({"role": "ai", "content": response})


            # Display the AI's response
            with st.chat_message("ai"):
                st.markdown(response)
                if detected_sentiment == 0:
                    st.session_state.user_sentiment = 'Negetive'
                elif detected_sentiment == 1:
                    st.session_state.user_sentiment = 'Neutral'
                elif detected_sentiment == 1:
                    st.session_state.user_sentiment = 'Positive'
                else:
                    st.session_state.user_sentiment = 'Neutral'
                
                # Convert the response to speech
                speech_fp = text_to_speech(response)
                # Play the speech
                st.audio(speech_fp, format='audio/mp3')
                

# Check if there are generated responses to display
with response_container:        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

                                                             
                            
with st.sidebar.expander("Sentiment Analysis"):
        # Use the values stored in session state
        st.write(
            f"- Detected User Tone: {st.session_state.user_sentiment}"
        )
            
        # Display Q-table
        st.dataframe(display_q_table(states))