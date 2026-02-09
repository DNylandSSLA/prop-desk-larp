"""Django settings for the Prop Desk web dashboard."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Add the repo root to sys.path so we can import prop_desk, mc_engine, bank_python
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECRET_KEY = "prop-desk-larp-not-for-production"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "daphne",
    "channels",
    "django.contrib.staticfiles",
    "desk_app",
]

# Channels
ASGI_APPLICATION = "desk_project.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}

ROOT_URLCONF = "desk_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "static" / "frontend"],
        "APP_DIRS": False,
        "OPTIONS": {
            "context_processors": [],
        },
    },
]

# Static files — Vite builds here
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]
STATIC_ROOT = BASE_DIR / "collected_static"

# No database needed
DATABASES = {}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Logging — keep it clean
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "brief": {
            "format": "%(asctime)s  %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "brief",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "desk_app": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        # Suppress noisy libs
        "bank_python": {"level": "WARNING"},
        "daphne": {"level": "WARNING"},
        "django": {"level": "WARNING"},
        "django.channels": {"level": "WARNING"},
    },
}
