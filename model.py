import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import Optional
from config import config

load_dotenv()

CACHE_DIR = config.models_dir


class ChatModel:
    def __init__(self, model_config):
        self.model_id = model_config.model_id
        self.device = model_config.device
        self.temperature = model_config.temperature
        self.top_p = model_config.top_p
        self.max_new_tokens = model_config.max_new_tokens
        
        ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        ).to(self.device)
        self.model.eval()
        self.chat_history = []
    
    def generate(self, question: str, context: Optional[str] = None, max_new_tokens: int = 512) -> str:
        """
        Generate response for a given question with optional context
        
        Args:
            question: User's question
            context: Optional context from RAG
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response string
        """
        prompt = self._build_prompt(question, context)
        
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            formatted_prompt, 
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        attention_mask = inputs['attention_mask']
        inputs = inputs['input_ids']
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self._clean_response(response, formatted_prompt)
    
    def _build_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build appropriate prompt based on whether context is provided"""
        if context is None or context.strip() == "":
            return f"""Give a detailed and helpful answer to the following question. 
            Be conversational and friendly in your response.
            
            Question: {question}"""
        else:
            return f"""
                    You are a helpful assistant.
                    You will be given a set of relevant documents and a user question.
                    Think through the problem step by step, referencing the documents as needed.
                    After reasoning carefully, provide a final concise answer.

                    Context Documents:
                    {context}

                    Question:
                    {question}

                    Let's think step by step:"""
    
    def _clean_response(self, response: str, formatted_prompt: str) -> str:
        """Clean and format the model response"""
        # Remove input prompt and special tokens from response
        response = response[len(formatted_prompt):]
        response = response.replace("<eos>", "").replace("</s>", "").replace("<|endoftext|>", "")
        
        # Clean up whitespace
        response = response.strip()
        
        # Remove any remaining artifacts
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and not line.endswith('>'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else response
    
    def generate_with_history(self, question: str, context: Optional[str] = None, 
                            max_new_tokens: int = None, max_history: int = None) -> str:
        """
        Generate response with conversation history consideration
        
        Args:
            question: Current user question
            context: Optional RAG context
            max_new_tokens: Maximum tokens to generate
            max_history: Maximum number of previous exchanges to consider
            
        Returns:
            Generated response
        """
        # Build conversation history
        max_history = max_history or config.get_agent_config().max_conversation_history
        recent_history = self.chat_history[-max_history:] if self.chat_history else []
        
        # Create conversation context
        conversation_context = ""
        if recent_history:
            conversation_context = "Previous conversation:\n"
            for exchange in recent_history:
                conversation_context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
            conversation_context += "\n"
        
        # Build current prompt
        if context:
            full_prompt = f"""{conversation_context}Using the provided document context, answer the current question.
            
            Document Context: {context}
            
            Current Question: {question}"""
        else:
            full_prompt = f"""{conversation_context}Answer the current question helpfully and conversationally.
            
            Current Question: {question}"""
        
        response = self.generate(full_prompt, max_new_tokens=max_new_tokens)
        self.chat_history.append({
            'user': question,
            'assistant': response
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_id': self.model_id,
            'device': self.device,
            'tokenizer_vocab_size': len(self.tokenizer),
            'chat_history_length': len(self.chat_history)
        }
