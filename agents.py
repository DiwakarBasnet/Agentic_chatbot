import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import dateparser
from dateparser.search import search_dates
from config import TIME_PATTERN, PHONE_PATTERN, EMAIL_PATTERN, CALL_PATTERNS, APPOINTMENT_PATTERNS


class ConversationState(Enum):
    NORMAL = "normal"
    COLLECTING_CONTACT = "collecting_contact"
    BOOKING_APPOINTMENT = "booking_appointment"


@dataclass
class UserInfo:
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    reason: Optional[str] = None


class InputValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        return re.match(EMAIL_PATTERN, email) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        cleaned_phone = re.sub(r'[\s\-\(\)\.]+', '', phone)
        return re.match(PHONE_PATTERN, cleaned_phone) is not None
    
    @staticmethod
    def validate_time(time_str: str) -> bool:   
        """Validate time format"""
        return re.match(TIME_PATTERN, time_str.strip()) is not None or ':' in time_str
    
    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Extract and convert date from natural language to YYYY-MM-DD format"""
        settings = {
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': datetime.now(),
            'RETURN_AS_TIMEZONE_AWARE': False
        }
        
        result = search_dates(text, settings=settings)
        
        if result:
            # result is a list of tuples (matched_text, datetime_obj)
            _, parsed_date = result[0]
            return parsed_date.strftime('%Y-%m-%d')
        
        return None


class IntentDetector:
    """Detect user intents from messages"""
    @classmethod
    def detect_intent(cls, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        if any(pattern in message_lower for pattern in CALL_PATTERNS):
            return 'request_callback'
        elif any(pattern in message_lower for pattern in APPOINTMENT_PATTERNS):
            return 'book_appointment'
        
        return 'general_query'


class ConversationalAgent:
    """Main conversational agent for handling multi-step forms"""
    
    def __init__(self):
        self.state = ConversationState.NORMAL
        self.user_info = UserInfo()
        self.current_field = None
        self.intent_detector = IntentDetector()
        
    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        return self.intent_detector.detect_intent(message)
    
    def process_contact_collection(self, message: str) -> Tuple[str, bool]:
        """Process contact information collection"""
        if self.current_field == 'name':
            self.user_info.name = message.strip()
            self.current_field = 'phone'
            return "Great! Now, please provide your phone number:", False
        
        elif self.current_field == 'phone':
            if InputValidator.validate_phone(message):
                self.user_info.phone = message.strip()
                self.current_field = 'email'
                return "Perfect! Finally, please provide your email address:", False
            else:
                return "Please provide a valid phone number (e.g., 9841248772):", False
        
        elif self.current_field == 'email':
            if InputValidator.validate_email(message):
                self.user_info.email = message.strip()
                response = self._format_contact_confirmation()
                self.reset_state()
                return response, True
            else:
                return "Please provide a valid email address (e.g., user@example.com):", False
        
        return "Something went wrong. Let's start over.", True
    
    def process_appointment_booking(self, message: str) -> Tuple[str, bool]:
        """Process appointment booking"""
        if self.current_field == 'name':
            self.user_info.name = message.strip()
            self.current_field = 'phone'
            return "Thank you! Please provide your phone number:", False
        
        elif self.current_field == 'phone':
            if InputValidator.validate_phone(message):
                self.user_info.phone = message.strip()
                self.current_field = 'email'
                return "Great! Now please provide your email address:", False
            else:
                return "Please provide a valid phone number:", False
        
        elif self.current_field == 'email':
            if InputValidator.validate_email(message):
                self.user_info.email = message.strip()
                self.current_field = 'date'
                return "Perfect! When would you like to schedule the appointment? ", False
            else:
                return "Please provide a valid email address:", False
        
        elif self.current_field == 'date':
            extracted_date = InputValidator.extract_date(message)
            if extracted_date:
                self.user_info.appointment_date = extracted_date
                self.current_field = 'time'
                return f"Great! Date set for {extracted_date}. What time would you prefer? ", False
            else:
                return "I couldn't understand the date. Please provide a clearer date (e.g., 'next Monday', 'tomorrow', or '2024-12-25'):", False
        
        elif self.current_field == 'time':
            if InputValidator.validate_time(message):
                self.user_info.appointment_time = message.strip()
                self.current_field = 'reason'
                return "Excellent! Finally, please briefly describe the reason for your appointment:", False
            else:
                return "Please provide a valid time format (e.g., '10:00 AM', '14:30', or '2:30 PM'):", False
        
        elif self.current_field == 'reason':
            self.user_info.reason = message.strip()
            response = self._format_appointment_confirmation()
            self.reset_state()
            return response, True
        
        return "Something went wrong. Let's start over.", True
    
    def start_contact_collection(self) -> str:
        """Start collecting contact information"""
        self.state = ConversationState.COLLECTING_CONTACT
        self.current_field = 'name'
        return "I'd be happy to have someone contact you! Let me collect your information.\n\nFirst, please provide your full name:"
    
    def start_appointment_booking(self) -> str:
        """Start appointment booking process"""
        self.state = ConversationState.BOOKING_APPOINTMENT
        self.current_field = 'name'
        return "I'll help you book an appointment! Let me collect some information.\n\nFirst, please provide your full name:"
    
    def reset_state(self):
        """Reset conversation state"""
        self.state = ConversationState.NORMAL
        self.user_info = UserInfo()
        self.current_field = None
    
    def _format_contact_confirmation(self) -> str:
        """Format contact information confirmation message"""
        return f"""Thank you! I have collected your information:
                                    
                    📞 **Contact Details Saved:**
                    - Name: {self.user_info.name}
                    - Phone: {self.user_info.phone}
                    - Email: {self.user_info.email}

                    Someone from our team will contact you shortly!"""
    
    def _format_appointment_confirmation(self) -> str:
        """Format appointment confirmation message"""
        return f"""**Appointment Booked Successfully!**

                    📅 **Appointment Details:**
                    - Name: {self.user_info.name}
                    - Phone: {self.user_info.phone}
                    - Email: {self.user_info.email}
                    - Date: {self.user_info.appointment_date}
                    - Time: {self.user_info.appointment_time}
                    - Reason: {self.user_info.reason}

                    You will receive a confirmation email shortly. Thank you!"""
    
    def get_current_state_info(self) -> dict:
        """Get current conversation state information for debugging"""
        return {
            'state': self.state.value,
            'current_field': self.current_field,
            'collected_info': {
                'name': self.user_info.name,
                'phone': self.user_info.phone,
                'email': self.user_info.email,
                'appointment_date': self.user_info.appointment_date,
                'appointment_time': self.user_info.appointment_time,
                'reason': self.user_info.reason
            }
        }