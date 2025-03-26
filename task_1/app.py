import os
import tempfile
import streamlit as st

from videodb import SceneExtractionType
from videodb import connect, SearchType, IndexType

from llama_index.llms.openai import OpenAI
from llama_index.core import get_response_synthesizer
from llama_index.retrievers.videodb import VideoDBRetriever
from llama_index.core.response_synthesizers import ResponseMode

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    videodb_api_key = st.text_input("VideoDB API Key", key="videodb_api_key", type="password")

if not openai_api_key or not videodb_api_key:
    st.info("Please add your OpenAI and VideoDB API keys to continue.")
    st.stop()

conn = connect(api_key=videodb_api_key)
coll = conn.create_collection(
    name="MAINGAMES", description="Video Retrievers"
)

def split_spoken_visual_query(query):
    transformation_prompt = """
    Divide the following query into two distinct parts: one for spoken content and one for visual content. The spoken content should refer to any narration, dialogue, or verbal explanations and The visual content should refer to any images, videos, or graphical representations. Format the response strictly as:\nSpoken: <spoken_query>\nVisual: <visual_query>\n\nQuery: {query}
    """
    
    prompt = transformation_prompt.format(query=query)
    response = OpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key
    ).complete(prompt)
    
    divided_query = response.text.strip().split("\n")
    
    spoken_query = divided_query[0].replace("Spoken:", "").strip()
    scene_query = divided_query[1].replace("Visual:", "").strip()
    
    return spoken_query, scene_query

st.set_page_config(page_title="Chat with the YOUR docs", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with YOUR video")
st.header("Powered by LlamaIndex 💬🦙, OpenAI and VideoDB")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = []

if "video" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.video = None

def get_user_inputs():
    data_source = st.selectbox('Select Data Source', ['youtube', 'file'])

    if data_source == 'youtube':
        video_url = st.text_input('Video URL')
        video = coll.upload(url=video_url)
    else:
        uploaded_file = st.file_uploader("Choose a mp4/mp3 file", type=["mp4", "mp3"], accept_multiple_files=False)
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(uploaded_file.name)[-1]) as video_file:
                video_file.write(uploaded_file.read())
                video = coll.upload(file_path=video_file.name)

    if "video" in st.session_state.keys(): # Initialize the chat messages history
        if st.session_state.video is not None:
            st.session_state.video.delete()

    st.session_state.video = video
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about \"{video.name}\"!"}
    ]

@st.cache_resource(show_spinner=False)
def index_data():
    with st.spinner(text="Loading and indexing YOUR video – hang tight! This should take 5-8 minutes."):

        st.session_state.video.index_spoken_words()
        index_id = st.session_state.video.index_scenes(
            extraction_type=SceneExtractionType.shot_based,
            extraction_config={
                "frame_count": 5
            },
            prompt="Describe the scene in detail with specific timestamp",
        )

        spoken_retriever = VideoDBRetriever(
            collection=coll.id,
            video=st.session_state.video.id,
            search_type=SearchType.semantic,
            index_type=IndexType.spoken_word,
            score_threshold=0.1,
        )
        
        scene_retriever = VideoDBRetriever(
            collection=coll.id,
            video=st.session_state.video.id,
            search_type=SearchType.semantic,
            index_type=IndexType.scene,
            scene_index_id=index_id,
            score_threshold=0.1,
        )

        return spoken_retriever, scene_retriever

get_user_inputs()

if st.button('Load Data'):
    spoken_retriever, scene_retriever = index_data()  # Pass the additional arguments here

if "video" not in st.session_state.keys():
    prompt = st.chat_input("Your question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Retrieving knowledge...") as spinner:
            spoken_query, scene_query = split_spoken_visual_query(prompt)

            spinner.text("Fetching relevant text...")
            # Fetch relevant nodes for Spoken index
            nodes_spoken_index = spoken_retriever.retrieve(spoken_query)

            spinner.text("Fetching relevant scenes...")
            # Fetch relevant nodes for Scene index
            nodes_scene_index = scene_retriever.retrieve(scene_query)

            spinner.text("Generating response...")
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT
            )

            response = response_synthesizer.synthesize(
                prompt, nodes=nodes_scene_index + nodes_spoken_index
            )

            st.session_state.messages.append({"role": "assistant", "content": response.response})

    for message in st.session_state.messages:
        with st.chat_message(f"{message['role']}"):
            st.write(message["content"])
else:
    st.write("Please load the data to begin chatting.")