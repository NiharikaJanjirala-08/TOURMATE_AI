# ============================================================
# TOURMATE AI
# Multimodal AI Travel Assistant
# ============================================================

import os
import tempfile
from datetime import datetime

import streamlit as st


# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

# -------------------- OLLAMA --------------------

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except Exception:
    chat = None
    OLLAMA_AVAILABLE = False


# -------------------- PDF --------------------

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PdfReader = None
    PDF_AVAILABLE = False


# -------------------- PILLOW --------------------

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False


# -------------------- FOLIUM --------------------

try:
    import folium
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except Exception:
    folium = None
    st_folium = None
    MAP_AVAILABLE = False


# -------------------- FASTER WHISPER --------------------

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    WHISPER_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TourMate AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "TourMate AI"

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:1b"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "base"
)


# ============================================================
# DESTINATION DATABASE
# ============================================================

DESTINATIONS = {

    "Goa": {
        "emoji": "🌴",
        "country": "India",
        "description": (
            "Beautiful beaches, nightlife, Portuguese "
            "heritage, seafood and relaxing coastal views."
        ),
        "latitude": 15.2993,
        "longitude": 74.1240,
        "budget": "₹8,000 - ₹25,000",
        "best_time": "November - February"
    },

    "Manali": {
        "emoji": "🏔️",
        "country": "India",
        "description": (
            "Snow-covered mountains, valleys, adventure "
            "activities and beautiful landscapes."
        ),
        "latitude": 32.2432,
        "longitude": 77.1892,
        "budget": "₹10,000 - ₹30,000",
        "best_time": "March - June"
    },

    "Jaipur": {
        "emoji": "🏰",
        "country": "India",
        "description": (
            "Royal palaces, forts, history, culture, "
            "shopping and traditional food."
        ),
        "latitude": 26.9124,
        "longitude": 75.7873,
        "budget": "₹7,000 - ₹20,000",
        "best_time": "October - March"
    },

    "Kerala": {
        "emoji": "🌊",
        "country": "India",
        "description": (
            "Backwaters, beaches, hill stations, wildlife "
            "and beautiful natural scenery."
        ),
        "latitude": 10.8505,
        "longitude": 76.2711,
        "budget": "₹10,000 - ₹30,000",
        "best_time": "October - February"
    },

    "Hyderabad": {
        "emoji": "🏙️",
        "country": "India",
        "description": (
            "Historic monuments, Charminar, biryani, "
            "shopping, culture and modern city life."
        ),
        "latitude": 17.3850,
        "longitude": 78.4867,
        "budget": "₹5,000 - ₹15,000",
        "best_time": "October - February"
    },

    "Bengaluru": {
        "emoji": "🌆",
        "country": "India",
        "description": (
            "Technology, cafes, parks, nightlife and "
            "nearby weekend attractions."
        ),
        "latitude": 12.9716,
        "longitude": 77.5946,
        "budget": "₹6,000 - ₹20,000",
        "best_time": "October - February"
    },

    "Mumbai": {
        "emoji": "🌃",
        "country": "India",
        "description": (
            "Marine Drive, Bollywood, street food, "
            "shopping and exciting city life."
        ),
        "latitude": 19.0760,
        "longitude": 72.8777,
        "budget": "₹8,000 - ₹25,000",
        "best_time": "November - February"
    },

    "Delhi": {
        "emoji": "🏛️",
        "country": "India",
        "description": (
            "Historic monuments, museums, street food "
            "and rich Indian culture."
        ),
        "latitude": 28.6139,
        "longitude": 77.2090,
        "budget": "₹7,000 - ₹22,000",
        "best_time": "October - March"
    }
}


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {

        "page": "Home",

        "saved_trips": [],

        "chat_messages": [],

        "generated_itinerary": "",

        "selected_destination": "Goa",

        "image_result": "",

        "document_text": "",

        "document_name": "",

        "document_answer": "",

        "voice_transcript": "",

        "voice_answer": ""
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ai(
    prompt,
    system_prompt=None,
    images=None,
    model=None
):

    if not OLLAMA_AVAILABLE:

        return (
            "⚠️ Ollama is not available.\n\n"
            "Install the Python package with:\n\n"
            "```text\n"
            "python -m pip install ollama\n"
            "```\n\n"
            "Also make sure the Ollama application is running."
        )

    selected_model = model or OLLAMA_MODEL

    try:

        messages = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        user_message = {
            "role": "user",
            "content": prompt
        }

        if images:

            user_message["images"] = images

        messages.append(user_message)

        response = chat(
            model=selected_model,
            messages=messages
        )

        try:

            return response["message"]["content"]

        except Exception:

            pass

        try:

            return response.message.content

        except Exception:

            pass

        return "⚠️ The AI returned an empty response."

    except Exception as error:

        return (
            "⚠️ Ollama Error\n\n"
            f"{error}\n\n"
            f"Model: {selected_model}\n\n"
            "Make sure Ollama is running and the model "
            "is installed."
        )


# ============================================================
# WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():

    if not WHISPER_AVAILABLE:

        return None

    try:

        model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        return model

    except Exception:

        return None


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_bytes,
    extension=".wav"
):

    if not WHISPER_AVAILABLE:

        return (
            "",
            "Faster-Whisper is not installed."
        )

    if not audio_bytes:

        return (
            "",
            "No audio was received."
        )

    temporary_file = None

    try:

        if not extension.startswith("."):

            extension = "." + extension

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as file:

            file.write(audio_bytes)

            temporary_file = file.name

        model = load_whisper_model()

        if model is None:

            return (
                "",
                (
                    "Could not load Faster-Whisper.\n"
                    f"Model: {WHISPER_MODEL}"
                )
            )

        segments, info = model.transcribe(
            temporary_file
        )

        transcript_parts = []

        for segment in segments:

            text = segment.text.strip()

            if text:

                transcript_parts.append(text)

        transcript = " ".join(
            transcript_parts
        ).strip()

        if not transcript:

            return (
                "",
                "No speech could be detected."
            )

        return transcript, ""

    except Exception as error:

        return (
            "",
            f"Audio transcription error: {error}"
        )

    finally:

        if temporary_file:

            try:

                os.remove(temporary_file)

            except Exception:

                pass


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    if not PDF_AVAILABLE:

        return ""

    try:

        reader = PdfReader(
            uploaded_file
        )

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:

                    pages.append(text)

            except Exception:

                continue

        return "\n".join(
            pages
        ).strip()

    except Exception:

        return ""


# ============================================================
# DOCUMENT QUESTION ANSWERING
# ============================================================

def answer_document_question(
    question,
    document
):

    if not document:

        return (
            "Please upload a PDF document first."
        )

    if not question.strip():

        return (
            "Please enter a question."
        )

    # Keep prompt size reasonable
    context = document[:30000]

    prompt = f"""
You are TourMate AI's Travel Document Assistant.

Answer the user's question using only the
uploaded document.

Do not invent information.

If the answer is not present in the document,
say:

"I couldn't find that information in the
uploaded document."

DOCUMENT:

{context}

USER QUESTION:

{question}
"""

    return ask_ai(
        prompt,
        (
            "You answer questions using uploaded "
            "travel documents. Never invent facts."
        )
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_travel_image(
    uploaded_image
):

    if not OLLAMA_AVAILABLE:

        return (
            "Ollama is not available. "
            "Start Ollama and try again."
        )

    try:

        image_bytes = uploaded_image.getvalue()

        prompt = """
You are TourMate AI's travel image assistant.

Analyze the uploaded travel image carefully.

Return the following sections:

## Estimated Location

Give the most likely location only if
there is enough visual evidence.

## Country

Give the likely country.

## Landmark

Identify a landmark only when there is
enough evidence.

## Visual Clues

Explain what is actually visible in the image.

## Confidence

Choose:

Low
Medium
High

## Travel Tip

Give one useful travel tip.

IMPORTANT:

- Do not claim certainty.
- Do not invent landmarks.
- Do not invent signs.
- Do not invent cities.
- Do not pretend to know information that
  cannot be seen.
"""

        return ask_ai(
            prompt,
            (
                "You are a careful multimodal travel "
                "image analysis assistant. "
                "Never fabricate visual information."
            ),
            images=[image_bytes]
        )

    except Exception as error:

        return (
            "⚠️ Image analysis failed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# ITINERARY GENERATOR
# ============================================================

def generate_itinerary(
    destination,
    days,
    travelers,
    budget,
    interests,
    starting_city
):

    prompt = f"""
Create a personalized travel itinerary.

Destination:
{destination}

Starting City:
{starting_city}

Number of Days:
{days}

Number of Travelers:
{travelers}

Budget Category:
{budget}

Interests:
{interests}

Create a practical day-by-day itinerary.

For each day include:

Morning
Afternoon
Evening
Food suggestions
Transportation
Approximate spending

Also include:

Total estimated budget
Best time to visit
Things to carry
Safety tips
Important travel advice
"""

    return ask_ai(
        prompt,
        (
            "You are an expert travel planner. "
            "Create practical, organized and realistic "
            "travel itineraries."
        )
    )


# ============================================================
# TRAVEL CHAT
# ============================================================

def travel_chat(
    question
):

    prompt = f"""
You are TourMate AI.

The user asked:

{question}

Help the user with their travel question.

You can help with:

- Destinations
- Itineraries
- Hotels
- Food
- Transportation
- Budgets
- Attractions
- Packing
- Travel tips
- General travel questions

Give a clear and practical answer.
"""

    return ask_ai(
        prompt,
        (
            "You are a friendly and professional "
            "AI travel assistant."
        )
    )


# ============================================================
# VOICE QUESTION
# ============================================================

def answer_voice_question(
    transcript
):

    prompt = f"""
The user spoke to TourMate AI.

Voice transcript:

{transcript}

Answer their travel question.

Give practical and useful information.
"""

    return ask_ai(
        prompt,
        (
            "You are TourMate AI's voice travel "
            "assistant."
        )
    )


# ============================================================
# SAVE TRIP
# ============================================================

def save_trip(
    destination,
    itinerary
):

    if not itinerary:

        return

    st.session_state.saved_trips.append(
        {
            "destination": destination,
            "itinerary": itinerary,
            "saved_at": datetime.now().strftime(
                "%d %B %Y %I:%M %p"
            )
        }
    )


# ============================================================
# NAVIGATION
# ============================================================

def render_navigation():

    left, right = st.columns(
        [4, 6]
    )

    with left:

        st.title(
            "✈️ TourMate AI"
        )

        st.caption(
            "Your intelligent travel companion"
        )

    pages = [
        "Home",
        "Explore",
        "Planner",
        "Assistant",
        "Photos",
        "Documents",
        "Voice",
        "Map",
        "Saved"
    ]

    navigation_columns = st.columns(5)

    for index, page in enumerate(pages):

        with navigation_columns[
            index % 5
        ]:

            if st.button(
                page,
                key=f"navigation_{page}",
                use_container_width=True
            ):

                st.session_state.page = page

                st.rerun()

    st.divider()


# ============================================================
# HOME PAGE
# ============================================================

def render_home():

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.info(
        "✨  AI TRAVEL PLATFORM"
    )

    st.title(
        "🌍 Explore the world with TourMate AI"
    )

    st.subheader(
        "Your intelligent travel companion"
    )

    st.write(
        """
        An AI-powered travel platform designed to
        help you discover destinations, plan personalized
        journeys, analyze travel images, understand
        travel documents and interact using voice.
        """
    )

    st.write("")

    hero_left, hero_right = st.columns(2)

    with hero_left:

        if st.button(
            "✨ Start Planning",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "Planner"

            st.rerun()

    with hero_right:

        if st.button(
            "🌍 Explore Destinations",
            use_container_width=True
        ):

            st.session_state.page = "Explore"

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.header(
        "Everything you need for smarter travel"
    )

    st.write(
        "One intelligent platform for discovering, "
        "planning and experiencing your journey."
    )

    st.write("")

    feature_list = [

        (
            "🧠 AI Trip Planner",
            "Create personalized day-by-day itineraries "
            "based on destination, budget and interests."
        ),

        (
            "🗺️ Interactive Maps",
            "Explore travel destinations using an "
            "interactive map."
        ),

        (
            "📄 Document Assistant",
            "Upload travel PDFs and ask questions "
            "about their contents."
        ),

        (
            "🎙️ Voice Assistant",
            "Speak to TourMate AI and receive "
            "intelligent travel assistance."
        ),

        (
            "📸 AI Image Analysis",
            "Analyze travel photos and get possible "
            "location clues."
        ),

        (
            "💬 AI Travel Assistant",
            "Ask natural-language questions about "
            "destinations, food, budgets and more."
        )
    ]

    columns = st.columns(3)

    for index, feature in enumerate(
        feature_list
    ):

        with columns[index % 3]:

            with st.container(
                border=True
            ):

                st.subheader(
                    feature[0]
                )

                st.write(
                    feature[1]
                )

    st.write("")

    st.divider()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.header(
        "TourMate AI at a glance"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Destinations",
            "08+"
        )

    with s2:

        st.metric(
            "AI Features",
            "06"
        )

    with s3:

        st.metric(
            "Travel Assistance",
            "24/7"
        )

    with s4:

        st.metric(
            "Trip Possibilities",
            "∞"
        )

    st.write("")

    st.divider()

    # --------------------------------------------------------
    # WHY TOURMATE
    # --------------------------------------------------------

    st.header(
        "Why TourMate AI?"
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🌍 Discover"
        )

        st.write(
            "Find destinations based on your "
            "travel interests."
        )

        st.subheader(
            "🧠 Personalize"
        )

        st.write(
            "Generate personalized itineraries "
            "according to your budget and interests."
        )

        st.subheader(
            "📸 Understand"
        )

        st.write(
            "Use AI to analyze travel photographs "
            "and visual clues."
        )

    with right:

        st.subheader(
            "🎤 Speak"
        )

        st.write(
            "Use your voice to communicate naturally "
            "with TourMate AI."
        )

        st.subheader(
            "📄 Ask Documents"
        )

        st.write(
            "Upload travel documents and ask questions "
            "about their contents."
        )

        st.subheader(
            "🗺️ Explore"
        )

        st.write(
            "Explore destinations using interactive "
            "maps and location information."
        )

    st.divider()

    # --------------------------------------------------------
    # QUICK ACCESS
    # --------------------------------------------------------

    st.header(
        "🚀 Quick Access"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        if st.button(
            "✈️ Plan a Trip",
            use_container_width=True
        ):

            st.session_state.page = "Planner"

            st.rerun()

    with q2:

        if st.button(
            "💬 Ask AI",
            use_container_width=True
        ):

            st.session_state.page = "Assistant"

            st.rerun()

    with q3:

        if st.button(
            "📸 Analyze Photo",
            use_container_width=True
        ):

            st.session_state.page = "Photos"

            st.rerun()

    with q4:

        if st.button(
            "🎙️ Voice Assistant",
            use_container_width=True
        ):

            st.session_state.page = "Voice"

            st.rerun()


# ============================================================
# EXPLORE PAGE
# ============================================================

def render_explore():

    st.title(
        "🌍 Explore Destinations"
    )

    st.write(
        "Discover destinations and start planning "
        "your next journey."
    )

    for destination, data in DESTINATIONS.items():

        with st.container(
            border=True
        ):

            left, right = st.columns(
                [4, 1]
            )

            with left:

                st.subheader(
                    f"{data['emoji']} {destination}"
                )

                st.write(
                    data["description"]
                )

                st.write(
                    f"🌍 Country: {data['country']}"
                )

                st.write(
                    f"💰 Estimated Budget: {data['budget']}"
                )

                st.write(
                    f"📅 Best Time: {data['best_time']}"
                )

            with right:

                if st.button(
                    "Plan Trip",
                    key=f"plan_{destination}",
                    use_container_width=True
                ):

                    st.session_state.selected_destination = (
                        destination
                    )

                    st.session_state.page = "Planner"

                    st.rerun()


# ============================================================
# PLANNER PAGE
# ============================================================

def render_planner():

    st.title(
        "✈️ AI Trip Planner"
    )

    st.write(
        "Create a personalized travel itinerary "
        "using AI."
    )

    destination_names = list(
        DESTINATIONS.keys()
    )

    selected_destination = (
        st.session_state.selected_destination
    )

    if selected_destination not in destination_names:

        selected_destination = destination_names[0]

    destination = st.selectbox(
        "Destination",
        destination_names,
        index=destination_names.index(
            selected_destination
        )
    )

    st.session_state.selected_destination = (
        destination
    )

    left, right = st.columns(2)

    with left:

        starting_city = st.text_input(
            "Starting City",
            value="Hyderabad"
        )

        days = st.number_input(
            "Number of Days",
            min_value=1,
            max_value=30,
            value=4
        )

        travelers = st.number_input(
            "Number of Travelers",
            min_value=1,
            max_value=20,
            value=2
        )

    with right:

        budget = st.selectbox(
            "Budget",
            [
                "Budget",
                "Moderate",
                "Premium",
                "Luxury"
            ]
        )

        interests = st.multiselect(
            "Interests",
            [
                "Nature",
                "Adventure",
                "Beaches",
                "History",
                "Culture",
                "Food",
                "Shopping",
                "Photography",
                "Nightlife",
                "Relaxation"
            ],
            default=[
                "Nature",
                "Food"
            ]
        )

    st.write("")

    if st.button(
        "✨ Generate AI Itinerary",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "TourMate AI is planning your trip..."
        ):

            itinerary = generate_itinerary(
                destination=destination,
                days=days,
                travelers=travelers,
                budget=budget,
                interests=", ".join(
                    interests
                ),
                starting_city=starting_city
            )

        st.session_state.generated_itinerary = (
            itinerary
        )

    if st.session_state.generated_itinerary:

        st.divider()

        st.header(
            "🗓️ Your AI Itinerary"
        )

        st.markdown(
            st.session_state.generated_itinerary
        )

        st.write("")

        download_col, save_col = st.columns(2)

        with download_col:

            st.download_button(
                "📥 Download Itinerary",
                data=(
                    st.session_state.generated_itinerary
                ),
                file_name="tourmate_itinerary.txt",
                mime="text/plain",
                use_container_width=True
            )

        with save_col:

            if st.button(
                "💾 Save This Trip",
                use_container_width=True
            ):

                save_trip(
                    destination,
                    st.session_state.generated_itinerary
                )

                st.success(
                    "Trip saved successfully!"
                )


# ============================================================
# CHAT ASSISTANT PAGE
# ============================================================

def render_assistant():

    st.title(
        "💬 TourMate AI Assistant"
    )

    st.write(
        "Ask anything about travel."
    )

    clear_col, empty_col = st.columns(
        [2, 8]
    )

    with clear_col:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_messages = []

            st.rerun()

    for message in (
        st.session_state.chat_messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask TourMate AI..."
    )

    if question:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "TourMate AI is thinking..."
            ):

                answer = travel_chat(
                    question
                )

            st.markdown(
                answer
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# PHOTO PAGE
# ============================================================

def render_photos():

    st.title(
        "📸 AI Travel Image Analyzer"
    )

    st.write(
        "Upload a travel photograph and analyze "
        "it using AI."
    )

    if not PIL_AVAILABLE:

        st.error(
            "Pillow is not installed."
        )

        st.code(
            "python -m pip install pillow"
        )

        return

    uploaded_image = st.file_uploader(
        "Upload Travel Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_image:

        try:

            image = Image.open(
                uploaded_image
            )

            st.image(
                image,
                caption="Uploaded Travel Image",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Could not open image: {error}"
            )

            return

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "TourMate AI is analyzing the image..."
            ):

                result = analyze_travel_image(
                    uploaded_image
                )

            st.session_state.image_result = (
                result
            )

    if st.session_state.image_result:

        st.divider()

        st.subheader(
            "🤖 AI Image Analysis"
        )

        st.markdown(
            st.session_state.image_result
        )

        st.warning(
            "⚠️ AI image location identification "
            "is an estimate and may be incorrect. "
            "Please verify the location before "
            "using it for travel decisions."
        )


# ============================================================
# DOCUMENT PAGE
# ============================================================

def render_documents():

    st.title(
        "📄 Travel Document Assistant"
    )

    st.write(
        "Upload a travel PDF and ask questions "
        "about its contents."
    )

    if not PDF_AVAILABLE:

        st.error(
            "pypdf is not installed."
        )

        st.code(
            "python -m pip install pypdf"
        )

        return

    pdf_file = st.file_uploader(
        "Upload Travel PDF",
        type=["pdf"]
    )

    if pdf_file:

        with st.spinner(
            "Reading PDF..."
        ):

            extracted_text = extract_pdf_text(
                pdf_file
            )

        if extracted_text:

            st.session_state.document_text = (
                extracted_text
            )

            st.session_state.document_name = (
                pdf_file.name
            )

            st.success(
                "PDF loaded successfully."
            )

            st.info(
                f"File: {pdf_file.name} | "
                f"Characters: {len(extracted_text):,}"
            )

        else:

            st.error(
                "No readable text was found "
                "in this PDF."
            )

    if st.session_state.document_text:

        st.divider()

        st.subheader(
            "📑 Uploaded Document"
        )

        st.write(
            st.session_state.document_name
        )

        question = st.text_input(
            "Ask a question about your PDF",
            placeholder=(
                "Example: What is the hotel "
                "check-in time?"
            )
        )

        if st.button(
            "🔎 Ask Document",
            type="primary",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Analyzing your document..."
                ):

                    answer = answer_document_question(
                        question,
                        st.session_state.document_text
                    )

                st.session_state.document_answer = (
                    answer
                )

        if st.session_state.document_answer:

            st.divider()

            st.subheader(
                "🤖 Answer"
            )

            st.markdown(
                st.session_state.document_answer
            )


# ============================================================
# VOICE PAGE
# ============================================================

def render_voice():

    st.title(
        "🎙️ Voice Travel Assistant"
    )

    st.write(
        "Talk to TourMate AI using your voice."
    )

    if not WHISPER_AVAILABLE:

        st.error(
            "Faster-Whisper is not installed."
        )

        st.code(
            "python -m pip install faster-whisper"
        )

        return

    # ========================================================
    # DIRECT RECORDING
    # ========================================================

    st.header(
        "🎙️ Record Assistant"
    )

    st.write(
        "Record your question directly in the browser."
    )

    try:

        recorded_audio = st.audio_input(
            "Click here to record your question"
        )

    except Exception as error:

        recorded_audio = None

        st.warning(
            f"Browser recording is unavailable: {error}"
        )

    if recorded_audio:

        st.audio(
            recorded_audio
        )

        if st.button(
            "🗣️ Process Recording",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Converting speech to text..."
            ):

                transcript, error = (
                    transcribe_audio(
                        recorded_audio.getvalue(),
                        ".wav"
                    )
                )

            if error:

                st.error(
                    error
                )

            else:

                st.session_state.voice_transcript = (
                    transcript
                )

                st.subheader(
                    "📝 Transcript"
                )

                st.write(
                    transcript
                )

                with st.spinner(
                    "TourMate AI is answering..."
                ):

                    answer = answer_voice_question(
                        transcript
                    )

                st.session_state.voice_answer = (
                    answer
                )

    # ========================================================
    # AUDIO UPLOAD
    # ========================================================

    st.divider()

    st.header(
        "🎧 Voice Upload Assistant"
    )

    st.write(
        "Upload a recorded voice file."
    )

    audio_file = st.file_uploader(
        "Upload Audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg"
        ],
        key="voice_upload"
    )

    if audio_file:

        st.audio(
            audio_file
        )

        if st.button(
            "🎤 Process Uploaded Audio",
            use_container_width=True
        ):

            extension = os.path.splitext(
                audio_file.name
            )[1]

            with st.spinner(
                "Transcribing audio..."
            ):

                transcript, error = (
                    transcribe_audio(
                        audio_file.getvalue(),
                        extension
                    )
                )

            if error:

                st.error(
                    error
                )

            else:

                st.session_state.voice_transcript = (
                    transcript
                )

                with st.spinner(
                    "TourMate AI is preparing an answer..."
                ):

                    st.session_state.voice_answer = (
                        answer_voice_question(
                            transcript
                        )
                    )

    # ========================================================
    # TRANSCRIPT
    # ========================================================

    if st.session_state.voice_transcript:

        st.divider()

        st.subheader(
            "📝 Voice Transcript"
        )

        st.write(
            st.session_state.voice_transcript
        )

    # ========================================================
    # ANSWER
    # ========================================================

    if st.session_state.voice_answer:

        st.subheader(
            "🤖 TourMate AI"
        )

        st.markdown(
            st.session_state.voice_answer
        )


# ============================================================
# MAP PAGE
# ============================================================

def render_map():

    st.title(
        "🗺️ Interactive Travel Map"
    )

    st.write(
        "Explore TourMate AI destinations."
    )

    if not MAP_AVAILABLE:

        st.error(
            "Folium is not installed."
        )

        st.code(
            "python -m pip install "
            "folium streamlit-folium"
        )

        return

    selected_destination = st.selectbox(
        "Select Destination",
        list(DESTINATIONS.keys())
    )

    selected_data = DESTINATIONS[
        selected_destination
    ]

    travel_map = folium.Map(
        location=[
            selected_data["latitude"],
            selected_data["longitude"]
        ],
        zoom_start=5
    )

    for name, data in DESTINATIONS.items():

        popup_text = (
            f"<b>{name}</b><br>"
            f"{data['description']}<br>"
            f"Budget: {data['budget']}<br>"
            f"Best time: {data['best_time']}"
        )

        folium.Marker(
            [
                data["latitude"],
                data["longitude"]
            ],
            tooltip=name,
            popup=popup_text
        ).add_to(
            travel_map
        )

    st_folium(
        travel_map,
        width=None,
        height=600
    )


# ============================================================
# SAVED TRIPS PAGE
# ============================================================

def render_saved():

    st.title(
        "💾 Saved Trips"
    )

    if not st.session_state.saved_trips:

        st.info(
            "No trips saved yet."
        )

        st.write(
            "Generate an itinerary in the "
            "AI Trip Planner and save it here."
        )

        return

    if st.button(
        "🗑️ Delete All Saved Trips"
    ):

        st.session_state.saved_trips = []

        st.success(
            "All saved trips have been deleted."
        )

        st.rerun()

    for index, trip in enumerate(
        st.session_state.saved_trips
    ):

        with st.expander(
            f"✈️ {trip['destination']} "
            f"— Saved {trip['saved_at']}"
        ):

            st.markdown(
                trip["itinerary"]
            )

            st.download_button(
                "📥 Download Itinerary",
                data=trip["itinerary"],
                file_name=(
                    f"{trip['destination']}_"
                    f"itinerary.txt"
                ),
                mime="text/plain",
                key=f"download_saved_{index}",
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.divider()

    st.caption(
        "✈️ TourMate AI"
    )

    st.caption(
        "An AI-powered travel assistant "
        "for smarter journey planning."
    )

    st.caption(
        "for smarter journey planning."
    )

    st.caption(
        "© 2026 TourMate AI"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

render_navigation()


current_page = st.session_state.page


if current_page == "Home":

    render_home()


elif current_page == "Explore":

    render_explore()


elif current_page == "Planner":

    render_planner()


elif current_page == "Assistant":

    render_assistant()


elif current_page == "Photos":

    render_photos()


elif current_page == "Documents":

    render_documents()


elif current_page == "Voice":

    render_voice()


elif current_page == "Map":

    render_map()


elif current_page == "Saved":

    render_saved()


else:

    st.session_state.page = "Home"

    render_home()


render_footer()