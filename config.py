from dotenv import load_dotenv
import os

load_dotenv()

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "info").upper()
PROJECT_NAME = os.getenv("PROJECT_NAME", "yolo-seg-whu")
MODEL = os.getenv("MODEL", "yolo26n-seg.pt")