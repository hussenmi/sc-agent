"""Terminal input helpers for SCAgent CLI sessions."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

_READLINE_CONFIGURED = False
_PROMPT_SESSION = None


@dataclass(frozen=True)
class DecisionChoice:
    """A user-facing decision label paired with a stable machine action."""

    label: str
    action: str
    requires_text: bool = False
    text_prompt: str = "Your response: "
    placeholder: str = ""


@dataclass(frozen=True)
class DecisionSelection:
    """Normalized result from either the selector or text fallback."""

    action: str
    label: str
    index: int | None
    value: str
    raw_response: str
    input_mode: str
    custom: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_ORDINALS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
}


def resolve_decision_response(
    response: str,
    choices: Sequence[DecisionChoice],
    *,
    default_index: int | None = None,
    allow_custom: bool = True,
    input_mode: str = "text",
) -> DecisionSelection:
    """Resolve a text reply against choices without making the model infer it."""

    raw = response or ""
    normalized = raw.strip()
    lowered = normalized.casefold()
    selected_index: int | None = None

    if not normalized and default_index is not None and 0 <= default_index < len(choices):
        selected_index = default_index
    else:
        number_match = re.fullmatch(r"(?:option|choice)?\s*#?\s*(\d+)", lowered)
        if number_match:
            selected_index = int(number_match.group(1)) - 1
        elif lowered in _ORDINALS:
            selected_index = _ORDINALS[lowered] - 1
        else:
            for index, choice in enumerate(choices):
                if lowered in {choice.action.casefold(), choice.label.casefold()}:
                    selected_index = index
                    break

    if selected_index is not None and 0 <= selected_index < len(choices):
        choice = choices[selected_index]
        custom = choice.action == "custom"
        return DecisionSelection(
            action=choice.action,
            label=choice.label,
            index=selected_index,
            value=normalized or choice.label,
            raw_response=raw,
            input_mode=input_mode,
            custom=custom,
        )

    if allow_custom:
        return DecisionSelection(
            action="custom",
            label="Custom response",
            index=None,
            value=normalized,
            raw_response=raw,
            input_mode=input_mode,
            custom=True,
        )

    return DecisionSelection(
        action="",
        label="",
        index=None,
        value=normalized,
        raw_response=raw,
        input_mode=input_mode,
        custom=False,
    )


def _run_selector_app(
    question: str,
    choices: Sequence[DecisionChoice],
    *,
    default_index: int = 0,
) -> int:
    """Render an arrow-key selector and return its zero-based choice index."""

    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.containers import Window

    selected = [min(max(default_index, 0), len(choices) - 1)]

    def content() -> FormattedText:
        fragments = [("class:question", f"{question}\n\n")]
        for index, choice in enumerate(choices):
            pointer = "❯" if index == selected[0] else " "
            style = "class:selected" if index == selected[0] else ""
            fragments.append((style, f" {pointer} {choice.label}\n"))
        fragments.append(("class:hint", "\n Use ↑/↓ and Enter."))
        return FormattedText(fragments)

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def _up(event):
        selected[0] = (selected[0] - 1) % len(choices)

    @bindings.add("down")
    @bindings.add("j")
    def _down(event):
        selected[0] = (selected[0] + 1) % len(choices)

    @bindings.add("home")
    def _home(event):
        selected[0] = 0

    @bindings.add("end")
    def _end(event):
        selected[0] = len(choices) - 1

    @bindings.add("enter")
    def _accept(event):
        event.app.exit(result=selected[0])

    @bindings.add("c-c")
    @bindings.add("escape")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    app = Application(
        layout=Layout(Window(FormattedTextControl(content), always_hide_cursor=True)),
        key_bindings=bindings,
        full_screen=False,
        erase_when_done=False,
    )
    return app.run()


def prompt_for_decision(
    question: str,
    choices: Sequence[DecisionChoice],
    *,
    default_index: int | None = 0,
    allow_custom: bool = True,
    custom_label: str = "Type something else...",
    custom_prompt: str = "Your response: ",
    custom_placeholder: str = "",
    force_text_fallback: bool = False,
    input_reader: Callable[..., str] | None = None,
) -> DecisionSelection:
    """Ask a discrete question using a selector, with a numbered text fallback."""

    input_reader = input_reader or read_user_input

    def read_text(prompt: str, placeholder: str = "") -> str:
        if input_reader is read_user_input:
            return input_reader(prompt, placeholder=placeholder)
        return input_reader(prompt)

    normalized_choices = list(choices)
    if not normalized_choices:
        response = read_text(f"{question}\n> ", custom_placeholder)
        return DecisionSelection(
            action="custom",
            label="Custom response",
            index=None,
            value=response,
            raw_response=response,
            input_mode="text",
            custom=True,
        )

    custom_index = next(
        (index for index, choice in enumerate(normalized_choices) if choice.action == "custom"),
        None,
    )
    if allow_custom and custom_index is None:
        custom_index = len(normalized_choices)
        normalized_choices.append(
            DecisionChoice(
                custom_label,
                "custom",
                requires_text=True,
                text_prompt=custom_prompt,
                placeholder=custom_placeholder,
            )
        )

    # Headless / non-interactive (e.g. NAT eval, batch jobs, CI): there is no TTY to
    # prompt on, so auto-select the default (recommended) option instead of blocking
    # on stdin. Prefer a concrete option over the synthetic "custom" text choice.
    # Only when using the real stdin reader — if a caller injected an input_reader
    # (tests / programmatic drivers), honor it instead.
    if input_reader is read_user_input and not sys.stdin.isatty():
        auto_index = default_index if (default_index is not None
                                       and 0 <= default_index < len(normalized_choices)) else 0
        if normalized_choices[auto_index].action == "custom":
            auto_index = next((i for i, c in enumerate(normalized_choices)
                               if c.action != "custom"), auto_index)
        choice = normalized_choices[auto_index]
        return DecisionSelection(
            action=choice.action,
            label=choice.label,
            index=auto_index,
            value=choice.label,
            raw_response=choice.label,
            input_mode="auto-default",
            custom=False,
        )

    use_selector = (
        not force_text_fallback
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    if use_selector:
        selected_index = _run_selector_app(
            question,
            normalized_choices,
            default_index=default_index or 0,
        )
        choice = normalized_choices[selected_index]
        if choice.requires_text or choice.action == "custom":
            response = read_text(choice.text_prompt, choice.placeholder)
            return DecisionSelection(
                action=choice.action,
                label=choice.label,
                index=selected_index,
                value=response,
                raw_response=response,
                input_mode="selector",
                custom=choice.action == "custom",
            )
        return DecisionSelection(
            action=choice.action,
            label=choice.label,
            index=selected_index,
            value=choice.label,
            raw_response=choice.label,
            input_mode="selector",
            custom=False,
        )

    print(question)
    for index, choice in enumerate(normalized_choices, 1):
        suffix = " [default]" if default_index == index - 1 else ""
        print(f"  {index}. {choice.label}{suffix}")
    response = read_text("> ")
    selection = resolve_decision_response(
        response,
        normalized_choices,
        default_index=default_index,
        allow_custom=allow_custom,
        input_mode="text",
    )
    selected_choice = (
        normalized_choices[selection.index]
        if selection.index is not None and 0 <= selection.index < len(normalized_choices)
        else None
    )
    if selected_choice is not None and (
        selected_choice.requires_text
        or (selection.action == "custom" and selection.index == custom_index)
    ):
        custom_response = read_text(
            selected_choice.text_prompt,
            selected_choice.placeholder,
        )
        return DecisionSelection(
            action=selected_choice.action,
            label=selection.label,
            index=selection.index,
            value=custom_response,
            raw_response=custom_response,
            input_mode="text",
            custom=selected_choice.action == "custom",
        )
    return selection


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
    """Create a prompt_toolkit session that supports safe multi-line paste.

    prompt_toolkit understands bracketed paste directly. With ``multiline=True``,
    pasted newlines are inserted into the buffer instead of being interpreted as
    separate submissions. The custom Enter binding accepts the whole buffer, so
    the normal single-line flow still feels like pressing Enter to send.
    """
    global _PROMPT_SESSION
    if _PROMPT_SESSION is not None:
        return _PROMPT_SESSION

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return None

    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _(event):
        # First Ctrl+C clears the buffer; pressing it again on an already
        # empty buffer aborts (raises KeyboardInterrupt for the caller).
        buf = event.current_buffer
        if buf.text:
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt)

    _PROMPT_SESSION = PromptSession(
        multiline=True,
        key_bindings=bindings,
        prompt_continuation="",
    )
    return _PROMPT_SESSION


def read_user_input(
    prompt: str = "",
    *,
    strip: bool = True,
    placeholder: str = "",
) -> str:
    """Read a line (or a full multi-line paste) from the user.

    When prompt_toolkit is available, pasted multi-line content is inserted into
    one editable prompt buffer and a normal Enter submits the whole request.

    If bracketed paste is not supported by the active terminal, type ``:paste``
    first. The helper then reads lines until a line containing only ``:end`` or
    ``\"\"\"`` and returns the whole block as one prompt.
    """
    session = _get_prompt_session()
    if session is not None:
        # Render the placeholder dimmed (grey italic) so it reads as a hint, not
        # as pre-filled input — prompt_toolkit shows a bare str in the normal
        # foreground, which looks like text the user already typed.
        ph: object = placeholder
        if placeholder:
            from prompt_toolkit.formatted_text import FormattedText

            ph = FormattedText([("italic fg:ansibrightblack", placeholder)])
        response = session.prompt(prompt, placeholder=ph)
    else:
        _configure_readline()
        if placeholder:
            print(f"[{placeholder}]")
        response = input(prompt)

    # Normalise line endings (\r\n from some terminals/clipboard managers → \n)
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    if response.strip() in {":paste", '"""'}:
        print("Paste mode — paste your full prompt, then finish with a line containing only :end")
        lines = []
        while True:
            line = input("... ")
            line = line.replace("\r\n", "\n").replace("\r", "\n")
            if line.strip() in {":end", '"""'}:
                response = "\n".join(lines)
                break
            lines.append(line)
    return response.strip() if strip else response
