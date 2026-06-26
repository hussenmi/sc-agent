"""Bug A: the structured decision follow-up message echoes option labels (e.g.
"Integrate the samples with scVI"). _remember_user_preferences keyword-scraped
that message and overwrote the user's actual multi_sample_strategy choice. It
must skip structured decision messages (the choice is already committed by
resolve_pending_decision) while still capturing genuine free-form user requests."""

from scagent.agent.agent import SCAgent
from scagent.agent.world_state import AgentWorldState


def _agent():
    a = object.__new__(SCAgent)
    a.world_state = AgentWorldState()
    a.run_manager = None
    return a


def test_structured_decision_message_does_not_clobber_strategy():
    a = _agent()
    # The authoritative choice (investigate) is already committed.
    a.world_state.resolve_decision("multi_sample_strategy", "investigate_integration", source="user")
    # The follow-up message echoes the option label that previously mis-matched.
    msg = (
        "[Structured user decision]\n"
        '{\n  "selected_action": "investigate_integration",\n'
        '  "question": "How should I handle these samples? Integrate the samples with scVI; '
        'keep combined; analyze separately."\n}\n\n'
        "Treat selected_action as authoritative."
    )
    a._remember_user_preferences(msg)
    assert a.world_state.get_confirmed_value("multi_sample_strategy") == "investigate_integration"


def test_genuine_user_integrate_request_still_captured():
    # A real free-form user instruction must still be captured.
    a = _agent()
    a._remember_user_preferences("please integrate the samples with scVI")
    assert a.world_state.get_confirmed_value("multi_sample_strategy") == "integrate_scvi"
