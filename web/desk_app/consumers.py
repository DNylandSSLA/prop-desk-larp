import json
import logging

from channels.generic.websocket import AsyncJsonWebsocketConsumer

logger = logging.getLogger("desk_app")

SIMULATION_GROUP = "simulation"


class SimulationConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer that streams simulation state to clients."""

    async def connect(self):
        await self.channel_layer.group_add(SIMULATION_GROUP, self.channel_name)
        await self.accept()
        logger.info(f"Client connected: {self.channel_name}")

        # Send current snapshot immediately
        from desk_app.simulation import SimulationEngine
        engine = SimulationEngine.instance()
        snapshot = engine.get_snapshot()
        if snapshot:
            await self.send_json(snapshot)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(SIMULATION_GROUP, self.channel_name)
        logger.info(f"Client disconnected: {self.channel_name}")

    async def simulation_tick(self, event):
        """Handle simulation.tick messages from the group."""
        await self.send_json(event["data"])
