from dotenv import load_dotenv
import os

load_dotenv()

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "info").upper()
PROJECT_NAME = os.getenv("PROJECT_NAME", "yolo26s-seg-training")
MODEL = os.getenv("MODEL", "yolo26s-seg.pt")