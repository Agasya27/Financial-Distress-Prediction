"""Load project environment files in a consistent order (Streamlit + OpenRouter)."""

from __future__ import annotations

import os


def load_financial_distress_dotenv(project_root: str) -> str | None:
    """
    Load variables from, in order:
      1. `.env`
      2. `.env.local`
      3. `.env.example` — **only if** `OPENROUTER_API_KEY` is still unset (dev convenience).

    Returns a user-facing warning when step 3 supplied the key (so you can move it to `.env`).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    env_p = os.path.join(project_root, ".env")
    loc_p = os.path.join(project_root, ".env.local")
    ex_p = os.path.join(project_root, ".env.example")

    # override=True: values in `.env` win over empty/mistaken shell exports (common OpenRouter issue).
    load_dotenv(env_p, override=True)
    load_dotenv(loc_p, override=True)

    def _key() -> str:
        return (os.getenv("OPENROUTER_API_KEY") or "").strip()

    if _key():
        return None

    if os.path.isfile(ex_p):
        load_dotenv(ex_p, override=True)
        if _key():
            return (
                "OpenRouter key was loaded from `.env.example`. "
                "Create a file named **`.env`** in this folder, paste the key there, "
                "and remove the real key from `.env.example` so it is never committed."
            )
    return None


def _manual_parse_openrouter_key(path: str) -> str | None:
    """Line-based parse (UTF-8 BOM safe) when `dotenv_values` misbehaves or python-dotenv is old."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, rest = line.partition("=")
                if key.strip() != "OPENROUTER_API_KEY":
                    continue
                val = rest.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                val = val.strip()
                if val:
                    return val
    except OSError:
        return None
    return None


def read_openrouter_key_from_dotenv_files(project_root: str) -> str | None:
    """
    Parse OPENROUTER_API_KEY directly from `.env` files on disk.

    Tries a small manual parser first, then `dotenv_values`, so Streamlit / tooling quirks
    and BOM / spacing issues still resolve the key.
    """
    for name in (".env", ".env.local", ".env.example"):
        path = os.path.join(project_root, name)
        if not os.path.isfile(path):
            continue
        v = _manual_parse_openrouter_key(path)
        if v:
            return v
        try:
            from dotenv import dotenv_values

            data = dotenv_values(path)
            v2 = (data.get("OPENROUTER_API_KEY") or "").strip()
            if v2:
                return v2
        except ImportError:
            continue
        except OSError:
            continue
    return None
