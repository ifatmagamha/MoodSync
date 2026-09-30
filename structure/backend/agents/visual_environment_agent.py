from phue import Bridge

# Calibration table: CIE xy colour, brightness (1-254), and a hex swatch for the dashboard preview.
# Tune per room: bulbs differ in gamut, and the same xy looks different in daylight.
LIGHT_SCENES = {
    "calm": {"xy": [0.22, 0.24], "bri": 110, "hex": "#8fb4e3"},         # soft blue, dimmed
    "uplifting": {"xy": [0.45, 0.41], "bri": 220, "hex": "#ffd98a"},    # bright warm daylight
    "grounding": {"xy": [0.53, 0.41], "bri": 90, "hex": "#f0a868"},     # low amber
    "neutral": {"xy": [0.3805, 0.3768], "bri": 180, "hex": "#fff1dc"},  # 4000K white
}


class VisualLightAgent:
    def __init__(self, bridge_ip=None, group_id=0):
        """
        bridge_ip=None is preview mode: scenes are returned but no light is touched.
        group_id is the Hue room/zone; 0 means every light on the bridge.
        """
        self.bridge = Bridge(bridge_ip) if bridge_ip else None  # first connect needs the bridge's link button
        self.group_id = group_id

    def adjust_lighting(self, emotional_state, transition_seconds=4):
        """Fade the room to the scene for this state. Returns the scene applied."""
        scene = LIGHT_SCENES.get(emotional_state, LIGHT_SCENES["neutral"])
        if self.bridge:
            self.bridge.set_group(self.group_id, {"on": True, "xy": scene["xy"], "bri": scene["bri"],
                                                  "transitiontime": transition_seconds * 10})
        return scene
