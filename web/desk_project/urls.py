from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = []

# Static files FIRST (before catch-all)
if settings.DEBUG:
    urlpatterns += [
        re_path(r'^static/(?P<path>.*)$', serve, {
            'document_root': settings.STATICFILES_DIRS[0],
        }),
    ]

urlpatterns += [
    path("", include("desk_app.urls")),
]
