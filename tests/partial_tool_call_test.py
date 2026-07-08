"""When a response truncates mid-tool-call, the partial tool-call fragment leaks
into the content channel (e.g. "<tool_call>write_json<arg_key>…"). The length
branch must detect that and suppress the raw dump instead of printing a wall of
broken JSON. This pins the detector."""

from scagent.agent.agent import SCAgent


def test_detects_partial_glm_tool_call():
    text = (
        "Building the annotation evidence for all 35 clusters now."
        '<tool_call>write_json<arg_key>name</arg_key><arg_value>annotation_evidence'
        '</arg_value><arg_key>data</arg_key><arg_value>{"0":{"label":"Alveolar'
    )
    assert SCAgent._looks_like_partial_tool_call(text) is True


def test_detects_xml_tool_call_markers():
    for marker in ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>", "<tools>"):
        assert SCAgent._looks_like_partial_tool_call(f"some text {marker} more") is True


def test_normal_narration_is_not_flagged():
    text = (
        "Loaded and explored the data. 43,632 cells x 33,694 genes, raw counts, "
        "8 donors. This is the Reyfman 2018 human lung dataset; proceeding to QC."
    )
    assert SCAgent._looks_like_partial_tool_call(text) is False


def test_empty_is_not_flagged():
    assert SCAgent._looks_like_partial_tool_call("") is False
    assert SCAgent._looks_like_partial_tool_call(None) is False
