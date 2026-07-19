"""A-share data archive V2: replaceable, resumable, integrity-proven."""

from .config import ArchiveConfig
from .pipeline import ArchivePipeline, EndpointSpec
from .probe import run_permission_probe
from .provider import (
    ArchiveProvider,
    ArchiveResponse,
    MockArchiveProvider,
    TushareCompatibleHttpProvider,
)
from .registry import EndpointInventory, default_inventory
from .sample import run_phase_a_sample
from .state import DownloadTask, TaskStateDB, TaskStatus

__all__ = [
    "ArchiveConfig",
    "ArchivePipeline",
    "ArchiveProvider",
    "ArchiveResponse",
    "DownloadTask",
    "EndpointInventory",
    "EndpointSpec",
    "MockArchiveProvider",
    "TaskStateDB",
    "TaskStatus",
    "TushareCompatibleHttpProvider",
    "default_inventory",
    "run_permission_probe",
    "run_phase_a_sample",
]
