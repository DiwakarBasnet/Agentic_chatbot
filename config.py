import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the chat model"""
    model_id: str = "google/gemma-2b-it"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    chunk_size: int = 256
    chunk_overlap_ratio: float = 0.1
    retrieval_k: int = 3
    distance_strategy: str = "COSINE"


@dataclass
class StreamlitConfig:
    """Configuration for Streamlit app"""
    page_title: str = "Chatbot"
    page_icon: str = "🤖"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


@dataclass
class AgentConfig:
    """Configuration for conversational agents"""
    max_conversation_history: int = 5
    intent_confidence_threshold: float = 0.7
    enable_debug_mode: bool = False


class AppConfig:
    """Main application configuration"""
    
    def __init__(self):
        # Directory paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.files_dir = os.path.join(self.base_dir, "files")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        
        # Ensure directories exist
        for dir_path in [self.files_dir, self.models_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configuration objects
        self.model = ModelConfig(cache_dir=self.models_dir)
        self.embedding = EmbeddingConfig()
        self.rag = RAGConfig()
        self.streamlit = StreamlitConfig()
        self.agent = AgentConfig()
        
        # Environment variables
        self.huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        return self.embedding
    
    def get_rag_config(self) -> RAGConfig:
        """Get RAG configuration"""
        return self.rag
    
    def get_streamlit_config(self) -> StreamlitConfig:
        """Get Streamlit configuration"""
        return self.streamlit
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        return self.agent
    
    def update_config_from_env(self):
        """Update configuration from environment variables"""
        # Model configuration
        if os.getenv("MODEL_ID"):
            self.model.model_id = os.getenv("MODEL_ID")
        if os.getenv("DEVICE"):
            self.model.device = os.getenv("DEVICE")
        if os.getenv("MAX_NEW_TOKENS"):
            self.model.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS"))
        
        # Embedding configuration
        if os.getenv("EMBEDDING_MODEL"):
            self.embedding.model_name = os.getenv("EMBEDDING_MODEL")
        
        # RAG configuration
        if os.getenv("CHUNK_SIZE"):
            self.rag.chunk_size = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("RETRIEVAL_K"):
            self.rag.retrieval_k = int(os.getenv("RETRIEVAL_K"))
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "directories": {
                "base_dir": self.base_dir,
                "files_dir": self.files_dir,
                "models_dir": self.models_dir,
                "logs_dir": self.logs_dir
            },
            "model": {
                "model_id": self.model.model_id,
                "device": self.model.device,
                "max_new_tokens": self.model.max_new_tokens,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "device": self.embedding.device,
                "normalize_embeddings": self.embedding.normalize_embeddings
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap_ratio": self.rag.chunk_overlap_ratio,
                "retrieval_k": self.rag.retrieval_k,
                "distance_strategy": self.rag.distance_strategy
            },
            "streamlit": {
                "page_title": self.streamlit.page_title,
                "page_icon": self.streamlit.page_icon,
                "layout": self.streamlit.layout,
                "initial_sidebar_state": self.streamlit.initial_sidebar_state
            },
            "agent": {
                "max_conversation_history": self.agent.max_conversation_history,
                "intent_confidence_threshold": self.agent.intent_confidence_threshold,
                "enable_debug_mode": self.agent.enable_debug_mode
            }
        }


# Global configuration instance
config = AppConfig()
config.update_config_from_env()


# Validation patterns for agents
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
PHONE_PATTERN = r'^[\+]?[1-9]?\d{9,15}$'
TIME_PATTERN = r'^([0-1]?[0-9]|2[0-3])[:\s][0-5][0-9](\s?(AM|PM|am|pm))?$|^([1-9]|1[0-2])(\s?(AM|PM|am|pm))$'

# Intent detection patterns
CALL_PATTERNS = [
    'call me', 'contact me', 'reach out', 'get in touch', 'phone me',
    'give me a call', 'call back', 'callback', 'ring me'
]

APPOINTMENT_PATTERNS = [
    'book appointment', 'schedule', 'make appointment', 'book meeting',
    'set appointment', 'appointment booking', 'schedule meeting',
    'book a session', 'reserve time', 'make reservation', 'book an appointment'
]

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': config.log_level,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': config.log_level,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(config.logs_dir, 'chatbot.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': config.log_level,
            'propagate': False
        }
    }
}