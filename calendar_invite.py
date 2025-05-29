from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to your downloaded credentials
SERVICE_ACCOUNT_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/calendar']
CALENDAR_ID = 'primary'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build('calendar', 'v3', credentials=credentials)

def create_calendar_event(user_info):
    start_time = datetime.strptime(f"{user_info.appointment_date} {user_info.appointment_time}", "%Y-%m-%d %I:%M %p")
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
        'attendees': [
            {'email': user_info.email}
        ],
        'reminders': {
            'useDefault': True,
        },
    }

    created_event = service.events().insert(calendarId=CALENDAR_ID, body=event, sendUpdates='all').execute()
    return created_event.get('htmlLink')
