"""Graduate: prove when your small model is ready to take over.

graduate.core     trajectory schema v0.3, canonical actions, slices, session splits
graduate.ingest   converters from trace formats, audited repairs, ingest report
graduate.certify  statistics (anytime-valid tests, bounds, multiplicity) and the report card
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("graduate")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
