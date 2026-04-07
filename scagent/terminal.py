"""Terminal input helpers for SCAgent CLI sessions."""

from __future__ import annotations

_READLINE_CONFIGURED = False


def _configure_readline() -> None:
    """Enable basic line editing when Python readline is available."""
    global _READLINE_CONFIGURED
    if _READLINE_CONFIGURED:
        return
    _READLINE_CONFIGURED = True

    try:
        import readline  # noqa: F401
    except ImportError:
        return


def read_user_input(prompt: str = "", *, strip: bool = True) -> str:
    """Read input with terminal line editing enabled when possible."""
    _configure_readline()
    response = input(prompt)
    return response.strip() if strip else response
