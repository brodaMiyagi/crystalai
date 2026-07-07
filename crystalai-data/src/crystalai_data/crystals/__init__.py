"""Crystal-structure curation: ICSD + MP-20 into a single SQLite database.

See ``DATA_ROADMAP.md`` §1. Space groups are read from the source (CIF text /
MP-20 CSV) and crystal systems derived from them — no spglib, no re-analysis.
"""

from .api import CrystalDatabase

__all__ = ["CrystalDatabase"]
