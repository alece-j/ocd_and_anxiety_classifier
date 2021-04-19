## This code comes from the streamlit_lesson from Caroline modified to fit my needs here

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_page_config(
    page_icon='📖',
    initial_sidebar_state='expanded'
)

st.title('Anxiety and OCD')

st.write('Use the sidebar to select a page to view.')

page = st.sidebar.selectbox(
    'Page',
    ('About', 'Diagnostic Suggestion Tool')
)

if page == 'About':
    st.subheader('About this project')
    st.write('''
This is a Streamlit app that hosts a non-medical diagnostic 'suggestion' that uses Natural Langauge Processing to give the user a sense of whether they are more likely to have OCD or Anxiety Disorders.
The model was generated using 1200 posts each from r/OCD and r/Anxiety subreddits from reddit.com.
The best model I found to make this diagnostic tool was a Logistic Regression model with tfidf vectorization of the text.
Please keep in mind this is NOT A MEDICAL DIAGNOSTIC TOOL and does not replace genuine medical advice.
Our aim is that people would use this tool as a jump off point to do further research and ultimately to consult a medical professional.
    ''')

elif page == 'Diagnostic Suggestion Tool':
    st.subheader('Based on your response, are you more likely to have OCD or Anxiety?')

    st.write('''
Enter some text to make a prediction.
The text might visually cut off, but you can write up to 1,000 characters.
    ''')

    # filepath different b/c solution code
    with open('./models/ocd_pipe.pkl', 'rb') as pickle_in:
        pipe = pickle.load(pickle_in)

    your_text = st.text_input(
        label='Tell us about your internal mental environment on days/at times when you feel off-balance.',
        value='My internal mental environment is ...',
        max_chars=1000)
    if pipe.predict([your_text])[0] == 0:
        disorder = 'Anxiety'
    else:
        disorder = 'OCD'
    st.subheader('Results:')
    st.write(f'Based on your response you may be interested in doing further research into: {disorder}.')

    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(your_text)
    just_compound = polarity.get('compound')
    st.subheader('Sentiment:')
    st.write(f'Based on your response it seems: {just_compound} where 1 is fully positive, 0 is neutral and -1 is wholly negative. Obviously given the prompt you would expect a score less than 0.')
