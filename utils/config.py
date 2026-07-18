"""
utils/config.py
===============
Central configuration file for the Vision-Auth pipeline.
All tuneable parameters live here so no constants are scattered
across modules.
"""

from dataclasses import dataclass, field


@dataclass
class CaptureSettings:
    """Settings consumed by core.capture.VideoCapture."""
    source_id:       int | str = 0       # 0 = default webcam
    width:           int       = 640
    height:          int       = 480
    target_fps:      float     = 30.0
    buffer_size:     int       = 64      # rolling frame buffer depth
    auto_reconnect:  bool      = True
    reconnect_delay: float     = 2.0     # seconds between reconnect tries


@dataclass
class PipelineConfig:
    """Top-level config that aggregates all feature-level settings."""
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    log_level: str = "INFO"              # DEBUG | INFO | WARNING | ERROR


# Singleton default config — import and mutate as needed
DEFAULT_CONFIG = PipelineConfig()
