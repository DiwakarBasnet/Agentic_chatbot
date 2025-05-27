import os
import streamlit as st
from typing import Optional
import logging

# Import custom modules
from model import ChatModel
from rag_util import Encoder, FaissDb, DocumentProcessor, RAGPipeline
from agents import ConversationalAgent, ConversationState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
)

# Ensure files directory exists
os.makedirs(FILES_DIR, exist_ok=True)


class StreamlitChatApp:
    """Main Streamlit chat application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Chatbot", 
            page_icon="🤖", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "agent" not in st.session_state:
            st.session_state.agent = ConversationalAgent()
        if "rag_pipeline" not in st.session_state:
            st.session_state.rag_pipeline = None
        if "model_loaded" not in st.session_state:
            st.session_state.model_loaded = False
    
    @st.cache_resource
    def load_model(_self):
        """Load and cache the chat model"""
        try:
            model = ChatModel(model_id="google/gemma-2b-it", device="cpu")
            logger.info("Chat model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            st.error(f"Failed to load chat model: {e}")
            return None
    
    @st.cache_resource
    def load_rag_pipeline(_self, embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2"):
        """Load and cache the RAG pipeline"""
        try:
            pipeline = RAGPipeline(embedding_model=embedding_model, device="cpu")
            logger.info("RAG pipeline loaded successfully")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load RAG pipeline: {e}")
            st.error(f"Failed to load RAG pipeline: {e}")
            return None
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to disk and return file path"""
        try:
            file_path = os.path.join(FILES_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"File saved: {uploaded_file.name}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save file {uploaded_file.name}: {e}")
            st.error(f"Failed to save file: {e}")
            return ""
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model parameters
            max_new_tokens = st.number_input(
                "Max New Tokens", 
                min_value=128, 
                max_value=4096, 
                value=512,
                help="Maximum number of tokens to generate"
            )
            
            k = st.number_input(
                "Number of Retrieved Documents (k)", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="Number of similar documents to retrieve for context"
            )
            
            st.header("📄 Document Upload")
            uploaded_files = st.file_uploader(
                "Upload PDFs for context", 
                type=["PDF", "pdf"], 
                accept_multiple_files=True,
                help="Upload PDF documents to provide context for your questions"
            )
            
            # Process uploaded files
            if uploaded_files:
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = self.save_uploaded_file(uploaded_file)
                    if file_path:
                        file_paths.append(file_path)
                
                if file_paths:
                    with st.spinner("Processing documents..."):
                        rag_pipeline = self.load_rag_pipeline()
                        if rag_pipeline and rag_pipeline.setup_database(file_paths):
                            st.session_state.rag_pipeline = rag_pipeline
                            st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                            
                            # Show pipeline stats
                            stats = rag_pipeline.get_pipeline_stats()
                            with st.expander("📊 Document Statistics"):
                                st.json(stats)
                        else:
                            st.error("Failed to process documents")
            
            st.header("💡 Features")
            st.info("""
            **Available Commands:**
            - Ask questions about uploaded documents
            - Say "call me" to request a callback
            - Say "book appointment" to schedule a meeting
            - Use natural language for dates (e.g., "next Monday")
            """)
            
            # Debug information
            if st.checkbox("Show Debug Info"):
                st.header("🔍 Debug Information")
                agent_info = st.session_state.agent.get_current_state_info()
                st.json(agent_info)
            
            return max_new_tokens, k
    
    def render_chat_interface(self, model, max_new_tokens: int, k: int):
        """Render the main chat interface"""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything, request a callback, or book an appointment!"):
            self.handle_user_input(prompt, model, max_new_tokens, k)
    
    def handle_user_input(self, prompt: str, model, max_new_tokens: int, k: int):
        """Handle user input and generate response"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process message with agent
        with st.chat_message("assistant"):
            response = self.generate_response(prompt, model, max_new_tokens, k)
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def generate_response(self, prompt: str, model, max_new_tokens: int, k: int) -> str:
        """Generate response based on conversation state and user input"""
        agent = st.session_state.agent
        
        try:
            # Handle different conversation states
            if agent.state == ConversationState.COLLECTING_CONTACT:
                response, is_complete = agent.process_contact_collection(prompt)
                return response
            
            elif agent.state == ConversationState.BOOKING_APPOINTMENT:
                response, is_complete = agent.process_appointment_booking(prompt)
                return response
            
            else:
                # Normal conversation - detect intent
                intent = agent.detect_intent(prompt)
                
                if intent == 'request_callback':
                    return agent.start_contact_collection()
                
                elif intent == 'book_appointment':
                    return agent.start_appointment_booking()
                
                else:
                    # Regular RAG query
                    return self.handle_rag_query(prompt, model, max_new_tokens, k)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def handle_rag_query(self, prompt: str, model, max_new_tokens: int, k: int) -> str:
        """Handle regular RAG queries"""
        context = None
        
        # Get context from RAG pipeline if available
        if st.session_state.rag_pipeline:
            try:
                with st.spinner("Searching documents..."):
                    context = st.session_state.rag_pipeline.search(prompt, k=k)
            except Exception as e:
                logger.error(f"Error during RAG search: {e}")
                st.warning("Failed to search documents, providing general response.")
        
        # Generate response
        try:
            with st.spinner("Generating response..."):
                response = model.generate(
                    prompt, 
                    context=context, 
                    max_new_tokens=max_new_tokens
                )
            return response
        except Exception as e:
            logger.error(f"Error generating model response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def render_header(self):
        """Render the main header"""
        st.title("🤖 Enhanced RAG Chatbot Assistant")
        st.markdown("*Ask questions about your documents, request callbacks, or book appointments!*")
        
        # Show connection status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_status = "✅ Connected" if st.session_state.get('model_loaded', False) else "❌ Not Connected"
            st.metric("Chat Model", model_status)
        
        with col2:
            rag_status = "✅ Ready" if st.session_state.rag_pipeline else "⚠️ No Documents"
            st.metric("RAG System", rag_status)
        
        with col3:
            agent_status = st.session_state.agent.state.value.replace('_', ' ').title()
            st.metric("Agent State", agent_status)
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        
        # Load model
        model = self.load_model()
        if not model:
            st.error("Failed to load the chat model. Please check your configuration.")
            st.stop()
        
        st.session_state.model_loaded = True
        
        # Render sidebar and get configuration
        max_new_tokens, k = self.render_sidebar()
        
        # Render main chat interface
        self.render_chat_interface(model, max_new_tokens, k)


def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitChatApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application encountered an error: {e}")


if __name__ == "__main__":
    main()