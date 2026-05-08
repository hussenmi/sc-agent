"""Terminal input helpers for SCAgent CLI sessions."""

from __future__ import annotations

_READLINE_CONFIGURED = False
_PROMPT_SESSION = None


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


def _get_prompt_session():
    """Create a modern prompt_toolkit session for interactive prompts.

    prompt_toolkit understands bracketed paste directly. With ``multiline=True``,
    pasted newlines are inserted into the buffer instead of being interpreted as
    separate submissions. The custom Enter binding accepts the whole buffer, so
    the normal single-line flow still feels like pressing Enter to send. Long
    pasted prompts keep the cursor at the start so the terminal displays the
    beginning of the request instead of jumping to the tail. Ctrl-C clears the
    current draft instead of raising KeyboardInterrupt out of the whole session.
    """
    global _PROMPT_SESSION
    if _PROMPT_SESSION is not None:
        return _PROMPT_SESSION

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
    except ImportError:
        return None

    bindings = KeyBindings()

    def _is_long_multiline_paste(data: str, event) -> bool:
        line_count = data.count("\n") + 1
        try:
            visible_rows = event.app.output.get_size().rows
        except Exception:
            visible_rows = 24
        return "\n" in data and line_count >= max(12, visible_rows - 6)

    @bindings.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    try:
        @bindings.add("s-enter")
        def _(event):
            event.current_buffer.insert_text("\n")
    except ValueError:
        # Some prompt_toolkit versions/terminals cannot name Shift-Enter as a
        # distinct key. Ctrl-J remains the portable newline fallback.
        pass

    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _(event):
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            buffer.validate_and_handle()

    @bindings.add(Keys.BracketedPaste)
    def _(event):
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        buffer = event.current_buffer
        paste_start = buffer.cursor_position
        buffer.insert_text(data)
        if _is_long_multiline_paste(data, event):
            buffer.cursor_position = paste_start

    _PROMPT_SESSION = PromptSession(
        multiline=True,
        key_bindings=bindings,
        mouse_support=True,
        prompt_continuation="",
    )
    return _PROMPT_SESSION


def read_user_input(prompt: str = "", *, strip: bool = True) -> str:
    """Read a line (or a full multi-line paste) from the user.

    When prompt_toolkit is available, pasted multi-line content is inserted into
    one editable prompt buffer and a normal Enter submits the whole request.

    If bracketed paste is not supported by the active terminal, type ``:paste``
    first. The helper then reads lines until a line containing only ``:end`` or
    ``\"\"\"`` and returns the whole block as one prompt.
    """
    session = _get_prompt_session()
    if session is not None:
        response = session.prompt(prompt)
    else:
        _configure_readline()
        response = input(prompt)

    # Normalise line endings (\r\n from some terminals/clipboard managers → \n)
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    if response.strip() in {":paste", '"""'}:
        print("Paste mode — paste your full prompt, then finish with a line containing only :end")
        lines = []
        while True:
            line = input()
            line = line.replace("\r\n", "\n").replace("\r", "\n")
            if line.strip() in {":end", '"""'}:
                response = "\n".join(lines)
                break
            lines.append(line)
    return response.strip() if strip else response
