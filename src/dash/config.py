import os

ENVRIONMENT = os.environ.get("ENVIRONMENT", "dev")
DEBUG = ENVRIONMENT == "prod"
HOST = '0.0.0.0' if ENVRIONMENT == "prod" else 'localhost'
