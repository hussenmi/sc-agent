"""world_state.snapshot() returns live references (e.g. user_preferences), and the
whole manifest is re-serialized on every save. Without freezing, every stored
world-state snapshot would alias the latest state — making per-step history a lie
(this is what masked the multi_sample_strategy mis-record). append_world_state_
snapshot must deep-copy so each stored snapshot is a true point-in-time record."""

from scagent.agent.run_manager import RunManager


def test_world_state_snapshots_are_frozen(tmp_path):
    rm = RunManager(base_dir=str(tmp_path)).create()

    live = {"user_preferences": {"multi_sample_strategy": "investigate_integration"}}
    rm.append_world_state_snapshot(live)

    # Mutate the live state the way the agent would on the next decision, then snap again.
    live["user_preferences"]["multi_sample_strategy"] = "integrate_scvi"
    rm.append_world_state_snapshot(live)

    snaps = rm.manifest.world_state_snapshots
    assert len(snaps) == 2
    # The first stored snapshot must still reflect the value at the time it was taken.
    assert snaps[0]["snapshot"]["user_preferences"]["multi_sample_strategy"] == "investigate_integration"
    assert snaps[1]["snapshot"]["user_preferences"]["multi_sample_strategy"] == "integrate_scvi"
