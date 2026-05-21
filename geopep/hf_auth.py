"""HuggingFace token resolution: env var > config > interactive prompt."""

import os
import sys


_PLACEHOLDER_TOKENS = {"", "hf_your_token_here", "none", "null"}


def _looks_like_placeholder(token) -> bool:
    if not isinstance(token, str):
        return True
    return token.strip().lower() in _PLACEHOLDER_TOKENS


def resolve_hf_token(config: dict, prompt: bool = True) -> str:
    """Resolve a HuggingFace token from env, config, or interactive prompt.

    Resolution order:
        1. HF_TOKEN environment variable (if set and not a placeholder).
        2. config['huggingface']['token'] (if not a placeholder).
        3. Interactive prompt via input() if `prompt=True` and stdin is a tty.

    On success, the resolved value is written into os.environ['HF_TOKEN']
    (so downstream `huggingface_hub` / `esm` calls pick it up) and returned.

    Raises RuntimeError if no token is available and interactive resolution
    is not possible.
    """
    env_token = os.environ.get("HF_TOKEN", "")
    if not _looks_like_placeholder(env_token):
        return env_token.strip()

    cfg_token = (config.get("huggingface") or {}).get("token", "")
    if not _looks_like_placeholder(cfg_token):
        token = cfg_token.strip()
        os.environ["HF_TOKEN"] = token
        return token

    if prompt and sys.stdin.isatty():
        print()
        print("=" * 70)
        print("HuggingFace token required to download ESM3 (gated model).")
        print("Get one at: https://huggingface.co/settings/tokens")
        print("(Grant the token access to the EvolutionaryScale/esm3 repo.)")
        print("=" * 70)
        try:
            token = input("Enter your HuggingFace token (hf_...): ").strip()
        except EOFError:
            token = ""
        if token and not _looks_like_placeholder(token):
            os.environ["HF_TOKEN"] = token
            return token

    raise RuntimeError(
        "No HuggingFace token available. Either:\n"
        "  - set the HF_TOKEN environment variable\n"
        "  - fill in `huggingface.token` in your config.yaml\n"
        "  - run the script interactively (will prompt)"
    )
