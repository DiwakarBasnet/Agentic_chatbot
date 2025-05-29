from datetime import datetime, timedelta
from dateutil import parser
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re
from dateutil import parser
import pytz

# Path to your downloaded credentials
SERVICE_ACCOUNT_FILE = 'credentials/calendar_service_account.json'
SCOPES = ['https://www.googleapis.com/auth/calendar']
CALENDAR_ID = 'primary'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build('calendar', 'v3', credentials=credentials)

def normalize_time_string(time_str: str) -> str:
    time_str = time_str.strip().lower()
    match = re.match(r'^(\d{1,2})\s(\d{2})\s?(am|pm)$', time_str)
    if match:
        hour, minute, meridiem = match.groups()
        return f"{hour}:{minute} {meridiem}"
    return time_str

def create_calendar_event(user_info):
    normalized_time = normalize_time_string(user_info.appointment_time)
    start_time = parser.parse(f"{user_info.appointment_date} {normalized_time}")
   
    utc = pytz.UTC
    start_time = utc.localize(start_time)
    end_time = start_time + timedelta(minutes=30)

    event = {
        'summary': f'Appointment with {user_info.name}',
        'description': user_info.reason,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'UTC',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'UTC',
        },
        'reminders': {
            'useDefault': True,
        },
    }

    created_event = service.events().insert(
        calendarId=CALENDAR_ID,
        body=event
    ).execute()

    return created_event.get('htmlLink')
