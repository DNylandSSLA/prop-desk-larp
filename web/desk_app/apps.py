import logging

from django.apps import AppConfig

logger = logging.getLogger("desk_app")


class DeskAppConfig(AppConfig):
    name = "desk_app"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        import sys
        # Only start the simulation when running the ASGI server (daphne/uvicorn),
        # not during management commands like collectstatic
        if "daphne" in sys.modules or "uvicorn" in sys.modules or any(
            "runserver" in arg for arg in sys.argv
        ):
            from desk_app.simulation import SimulationEngine
            engine = SimulationEngine.instance()
            engine.start()
