import pytest
from utils.cot_parser import parse_cot, has_tag_debris

@pytest.mark.parametrize("text,expected", [
    ("<reasoning>The circle is red.</reasoning><final>It is on the left.</final>", ("The circle is red.", "It is on the left.")),
    ("<reasonING >Round shape.</reasonING ><final >Red circle.</final >", ("Round shape.", "Red circle.")),
    ("<reasonning>Round shape.</reasonning <final'>Red circle.", ("Round shape.", "Red circle.")),
    ("<reasoning>Round shape.</reasoning><final></final>", ("Round shape.", "")),
    ("<reasoning>Round shape.</reasoning>", ("Round shape.", "")),
    ("Round shape. Red circle.", ("Round shape.", "Red circle.")),
    ("Red circle.", ("", "Red circle.")),
    ("", ("", "")),
    (None, ("", "")),
])
def test_frozen_parser_contract(text, expected):
    result = parse_cot(text)
    assert result == expected
    assert not any(has_tag_debris(part) for part in result)
