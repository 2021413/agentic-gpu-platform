"""Internal API used by GPU workers, protected by service authentication.

Deliberately empty of re-exports: ``routes`` depends on the API's dependency
providers, which in turn reference ``auth`` for the authenticator type. Pulling
both into this ``__init__`` would make that a circular import. Import the
submodules directly.
"""

from __future__ import annotations
