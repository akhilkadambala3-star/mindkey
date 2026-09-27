"""The shared data-store client.

Two backends, one interface (every route imports ``supabase`` from here):

- Default: Supabase, configured by ``SUPABASE_URL`` and ``SUPABASE_SECRET_KEY``.
- ``MINDKEY_STORE=local``: a SQLite file (``MINDKEY_LOCAL_DB``, default
  ``backend/.local/mindkey.db``) with the same query surface. Use it to run
  the full pipeline offline; see ``local_store.py``.
"""

import os

from dotenv import load_dotenv

load_dotenv()

STORE = os.getenv("MINDKEY_STORE", "supabase").strip().lower()

if STORE == "local":
    from local_store import LocalClient, default_local_path

    supabase = LocalClient(os.getenv("MINDKEY_LOCAL_DB") or default_local_path())
else:
    from supabase import Client, create_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError(
            "Supabase environment variables are not configured. Set SUPABASE_URL "
            "and SUPABASE_SECRET_KEY, or run offline with MINDKEY_STORE=local."
        )

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
