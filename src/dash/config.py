import os

ENVRIONMENT = os.environ.get("ENVIRONMENT", "dev")
DEBUG = ENVRIONMENT == "dev"
HOST = '0.0.0.0' if ENVRIONMENT == "prod" else 'localhost'
API_URL = os.environ.get("API_URL", "http://localhost:5000/api")
