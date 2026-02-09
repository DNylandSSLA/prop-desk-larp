"""ASGI config — routes HTTP and WebSocket protocols."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "desk_project.settings")

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

django_asgi = get_asgi_application()

from desk_app.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": URLRouter(websocket_urlpatterns),
})
