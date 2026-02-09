from django.urls import path, re_path

from desk_app.views import health, index

urlpatterns = [
    path("api/health/", health),
    re_path(r"^(?!ws/|static/|assets/|api/).*$", index),
]
