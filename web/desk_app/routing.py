from django.urls import path

from desk_app.consumers import SimulationConsumer

websocket_urlpatterns = [
    path("ws/simulation/", SimulationConsumer.as_asgi()),
]
