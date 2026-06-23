
# app.py - Strategy B Backend
# NOTE: Designed for Vercel. Ensure global app variable.
import os
import io
import re
import json
import math
import time
import base64
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import requests

# Global Flask app for Vercel import
app = Flask(__name__)

# Logging setup
logger = logging.getLogger("strategy_b")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Utility: robust JSON response

def safe_json(obj: Any, default=""):
    try:
        return json.dumps(obj)
    except Exception:
        return json.dumps(default)

# Text Normalizer and Tokenizer
class Normalizer:
    def __init__(self, lower=True, strip_accents=True):
        self.lower = lower
        self.strip_accents = strip_accents

    def _strip_accents(self, text: str) -> str:
        try:
            import unicodedata
            return ''.join(ch for ch in unicodedata.normalize('NFD', text) if unicodedata.category(ch) != 'Mn')
        except Exception:
            return text

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        t = text
        if self.lower:
            t = t.lower()
        t = re.sub(r"[
