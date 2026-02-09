import json

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


def health(request):
    from desk_app.simulation import SimulationEngine
    engine = SimulationEngine.instance()
    return JsonResponse({
        "status": "ok",
        "running": engine.running,
        "tick": engine.tick_count,
    })


def index(request):
    """Serve the React SPA index.html (production build)."""
    from django.conf import settings
    index_path = settings.BASE_DIR / "static" / "frontend" / "index.html"
    if index_path.exists():
        return HttpResponse(index_path.read_text(), content_type="text/html")
    return HttpResponse(
        "<h1>Prop Desk Dashboard</h1>"
        "<p>Frontend not built. Run <code>cd web/frontend && npm run build</code></p>",
        content_type="text/html",
    )
