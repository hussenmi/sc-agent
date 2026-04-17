"""Terminal input helpers for SCAgent CLI sessions."""

from __future__ import annotations

_READLINE_CONFIGURED = False


def _configure_readline() -> None:
    """Enable line editing and bracketed paste mode when readline is available.

    Bracketed paste mode tells the terminal to wrap pasted content in escape
    markers (\x1b[200~ ... \x1b[201~). GNU readline 8.1+ recognises these and
    returns the entire paste — including embedded newlines — as a single input()
    call instead of submitting one line at a time. This fixes the common issue
    where pasting multi-line text causes each line to be submitted separately.
    """
    global _READLINE_CONFIGURED
    if _READLINE_CONFIGURED:
        return
    _READLINE_CONFIGURED = True

    try:
        import readline
        readline.parse_and_bind("set enable-bracketed-paste on")
    except ImportError:
        pass


def read_user_input(prompt: str = "", *, strip: bool = True) -> str:
    """Read a line (or a full multi-line paste) from the user.

    With bracketed paste mode active, pasting multiple lines returns them all
    as a single string with embedded newlines — the agent receives the full
    context in one message rather than having each line trigger a separate turn.
    """
    _configure_readline()
    response = input(prompt)
    # Normalise line endings (\r\n from some terminals/clipboard managers → \n)
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    return response.strip() if strip else response
