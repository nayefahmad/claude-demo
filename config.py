import os
from dotenv import load_dotenv

load_dotenv()

# Location
CITY = os.getenv("CITY", "San Francisco")
CITY_LAT = os.getenv("CITY_LAT", "37.7749")
CITY_LON = os.getenv("CITY_LON", "-122.4194")
COUNTRY_CODE = os.getenv("COUNTRY_CODE", "US")

# Email settings
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "")

# Event APIs
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY", "")
EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_TOKEN", "")

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "city-events-recommender")

# App
DB_PATH = os.getenv("DB_PATH", "city_events.db")
FEEDBACK_BASE_URL = os.getenv("FEEDBACK_BASE_URL", "http://localhost:5000")
API_PORT = int(os.getenv("API_PORT", "5000"))
WEEKLY_DAY = os.getenv("WEEKLY_DAY", "monday")   # Day to send digest
WEEKLY_HOUR = int(os.getenv("WEEKLY_HOUR", "9"))  # Hour (24h) to send digest
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
