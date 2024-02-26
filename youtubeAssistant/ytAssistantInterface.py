import streamlit as st #build U.I with Python
import youtubeAssistant as ytAssistant
import textwrap

#Display Title of the PAge
st.title=("YouTube Assistant")

#Get User Inputs
with st.sidebar:#Side Bar Element
    #Get Youtube URL from User
    with st.form(key="my_form"):
        youtube_url=st.sidebar.text_area(
            label="What is the YouTube video URL",
            max_chars=50 #set max number of characters
        )
        #Get User Prompt
        query=st.sidebar.text_area(
            label="Ask me about the video",
            max_chars=50,
            key="query"
        )

        #Submit Form Button
        submit_button=st.form_submit_button(label='submit')
    
#Inject Inputs into Chat GPT
if query and youtube_url:
    db=ytAssistant.createVectorDBfromURL(youtube_url) #create vector database from yt video url
    response, docs=ytAssistant.get_response_from_query(db,query) #pass the db and query into the LLM to get a response
    st.subheader("Answer:")
    st.text(textwrap.fill(response,width=80))

