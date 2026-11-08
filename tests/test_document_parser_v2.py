"""Parser v2 (optimization_5): English tables and headings; v1 stays as it was."""

from typing import Any

from document_parser import ChunkingConfig, DocumentParser, _pdf_table_markdown
from scripts.download_quality_corpus import MarkdownExtractor, TextExtractor

RFC_TABLE = """16.  Variable-Length Integer Encoding

   This means that integers are encoded on 1, 2, 4, or 8 bytes.  Table 4
   summarizes the encoding properties.

          +======+========+=============+=======================+
          | 2MSB | Length | Usable Bits | Range                 |
          +======+========+=============+=======================+
          | 00   | 1      | 6           | 0-63                  |
          +------+--------+-------------+-----------------------+
          | 11   | 8      | 62          | 0-4611686018427387903 |
          +------+--------+-------------+-----------------------+

                   Table 4: Summary of Integer Encodings

   An example of a decoding algorithm is shown in Appendix A.1.
"""

RFC_WRAPPED_CELLS = """   +======+=========+============================+
   | MSB0 | Version | Description                |
   +======+=========+============================+
   | 0    | 4       | The randomly or            |
   |      |         | pseudorandomly generated   |
   |      |         | version.                   |
   +------+---------+----------------------------+
   | 0    | 6       | Reordered Gregorian time-  |
   |      |         | based UUID.                |
   +------+---------+----------------------------+

                       Table 2: UUIDv8 Versions
"""

BIT_DIAGRAM = """ 0                   1
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          unix_ts_ms           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""


def v2(**chunking: Any) -> DocumentParser:
    return DocumentParser(ChunkingConfig(**chunking) if chunking else None, version="v2")


def tables(parsed: Any) -> list[Any]:
    return [e for e in parsed.elements if e.kind == "table"]


def test_rfc_grid_table_is_one_table_with_header_and_trailing_caption() -> None:
    parsed = v2().parse_text(RFC_TABLE)
    [table] = tables(parsed)
    assert table.text.splitlines() == [
        "| 2MSB | Length | Usable Bits | Range |",
        "| --- | --- | --- | --- |",
        "| 00 | 1 | 6 | 0-63 |",
        "| 11 | 8 | 62 | 0-4611686018427387903 |",
    ]
    assert table.metadata["caption"] == "Table 4: Summary of Integer Encodings"
    assert table.metadata["table_number"] == "4"
    assert table.section_path == ("16 Variable-Length Integer Encoding",)
    [chunk] = [c for c in parsed.chunks if "table" in c.metadata["kinds"]]
    assert "Table 4: Summary of Integer Encodings\n| 2MSB | Length" in chunk.text


def test_wrapped_grid_cells_join_into_one_row() -> None:
    [table] = tables(v2().parse_text(RFC_WRAPPED_CELLS))
    rows = table.text.splitlines()[2:]
    assert rows == [
        "| 0 | 4 | The randomly or pseudorandomly generated version. |",
        "| 0 | 6 | Reordered Gregorian time-based UUID. |",
    ]


def test_bit_diagram_is_not_a_grid_table() -> None:
    one = DocumentParser().parse_text(BIT_DIAGRAM)
    two = v2().parse_text(BIT_DIAGRAM)
    assert not any(e.metadata.get("grid") for e in two.elements)
    assert [e.text for e in tables(one)] == [e.text for e in tables(two)]
    # v1 also took both bit rulers ("0   1", "0 1 2 3 ...") for headings; v2 does not.
    assert [e.text for e in one.elements if e.kind == "heading"] == [
        "1",
        "1 2 3 4 5 6 7 8 9 0 1 2 3 4 5",
    ]
    assert not [e for e in two.elements if e.kind == "heading"]


def test_caption_and_header_repeat_in_every_table_chunk() -> None:
    rows = [f"| item{i} | value{i} |" for i in range(40)]
    parsed = v2(target_tokens=40, max_tokens=80, overlap_tokens=5).parse_text(
        "Table 2. Summary of secrets\n| Name | Value |\n| --- | --- |\n" + "\n".join(rows)
    )
    chunks = [c for c in parsed.chunks if "table" in c.metadata["kinds"]]
    assert len(chunks) > 1
    assert all("Table 2. Summary of secrets\n| Name | Value |" in c.text for c in chunks)
    assert all(sum(row in c.text for c in chunks) == 1 for row in rows)


def test_prose_mentioning_a_table_is_not_a_caption() -> None:
    parsed = v2().parse_text("Table 2 lists all of the versions.\n| A | B |\n| 1 | 2 |")
    assert [e.kind for e in parsed.elements] == ["paragraph", "table"]
    assert "caption" not in tables(parsed)[0].metadata


def test_russian_caption_unchanged() -> None:
    text = "Таблица 2 — Параметры\n| Name | Value |\n| --- | --- |\n| a | 1 |"
    one, two = DocumentParser().parse_text(text), v2().parse_text(text)
    assert tables(one)[0].metadata["table_number"] == tables(two)[0].metadata["table_number"] == "2"
    assert tables(two)[0].metadata["caption"] == "Таблица 2 — Параметры"


def test_numbered_lines_that_continue_a_sentence_are_not_headings() -> None:
    text = "\n".join(
        [
            "1 Область применения",
            "4.2 Давление",
            "16.  Variable-Length Integer Encoding",
            "15.4.3.  302 Found",
            "15 minutes inactivity",
            "48 through 51 of the UUID).",
            "3 Include a description of the authorized uses of the system.",
            "24 The system characteristics that impact security and privacy vary and include the "
            "system type and function in terms",
            "308 (Permanent Redirect)",
        ]
    )
    headings = [e.text for e in v2().parse_text(text).elements if e.kind == "heading"]
    assert headings == ["Область применения", "Давление", "Variable-Length Integer Encoding", "302 Found"]
    legacy = [e.text for e in DocumentParser().parse_text(text).elements if e.kind == "heading"]
    assert "minutes inactivity" in legacy  # v1 keeps its behaviour for earlier runs


def test_list_items_and_addresses_are_not_headings() -> None:
    text = "\n".join(
        [
            "13.  Packetization and Reliability",
            "   3.  A limited on-path attacker can cause an idle connection to be",
            "2 Procedures to facilitate the implementation of the system and communications",
            "protection policy and the associated controls;",
            "2 Intended system usage; and",
            "2. Intended system usage; and",
            "1. Attacks and indicators of potential attacks in accordance with the following",
            "monitoring objectives;",
            "100 Bureau Drive (Mail Stop 8930) Gaithersburg, MD 20899-8930",
            "9 Is reviewed and approved by [Assignment: organization-defined personnel or roles]",
            "3 Access authorizations (i.e., privileges) and [Assignment: organization-defined",
            "J. Saltzer and M. Schroeder, The Protection of Information in Computer",
            "3.20 SUPPLY CHAIN RISK MANAGEMENT ........................ 363",
            "   # Greg_100_ns = 0x1EC9414C232AB00 or 138648505420000000",
            "A. Glossary",
            "## Notations",
            "2 Нормативные ссылки",
            "В настоящем стандарте использованы ссылки.",
        ]
    )
    headings = [e.text for e in v2().parse_text(text).elements if e.kind == "heading"]
    assert headings == ["Packetization and Reliability", "Glossary", "Notations", "Нормативные ссылки"]


def test_control_and_enhancement_headings_nest_under_the_section() -> None:
    text = "\n".join(
        [
            "3.1 ACCESS CONTROL",
            "AC-7",
            "UNSUCCESSFUL LOGON ATTEMPTS",
            "Control: Enforce a limit of consecutive invalid logon attempts.",
            "(4) UNSUCCESSFUL LOGON ATTEMPTS | USE OF ALTERNATE AUTHENTICATION FACTOR",
            "Allow the use of alternative factors.",
            "AC-8",
            "SYSTEM USE NOTIFICATION",
            "Display a notification.",
            "AC-9",
            "Previous Logon Notification",
        ]
    )
    elements = v2().parse_text(text).elements
    paths = {e.text: e.section_path for e in elements if e.kind == "paragraph"}
    assert paths["Allow the use of alternative factors."] == (
        "3.1 ACCESS CONTROL",
        "AC-7 UNSUCCESSFUL LOGON ATTEMPTS",
        "(4) UNSUCCESSFUL LOGON ATTEMPTS | USE OF ALTERNATE AUTHENTICATION FACTOR",
    )
    assert paths["Display a notification."] == ("3.1 ACCESS CONTROL", "AC-8 SYSTEM USE NOTIFICATION")
    # a title-case summary row is not a control heading
    assert "AC-9" in paths and paths["AC-9"] == paths["Display a notification."]


def test_russian_hierarchy_and_footnote_unchanged_in_v2() -> None:
    parsed = v2().parse_text(
        "4 Требования\n4.2 Давление\n4.2.1 Проверка\nДопуск 5 [^1].\n[^1]: При 20 °C."
    )
    target = next(e for e in parsed.elements if e.text.startswith("Допуск"))
    assert target.section_path == ("4 Требования", "4.2 Давление", "4.2.1 Проверка")
    assert target.metadata["footnote_context"] == ("[^1]: При 20 °C.",)


def test_v1_output_is_unchanged_for_the_grid_table() -> None:
    legacy = DocumentParser().parse_text(RFC_TABLE)
    kinds = [e.kind for e in legacy.elements]
    assert kinds.count("table") == 3  # header and each row separated by the borders
    assert not any(e.kind == "caption" for e in legacy.elements)


def test_pdf_table_drops_layout_columns_and_fills_spanned_header() -> None:
    rows: list[list[str | None]] = [
        ["", "ID", "", "FAMILY", "", "ID", "FAMILY"],
        ["", "AC", "", "Access  Control", "", "PE", "Physical and\nEnvironmental Protection"],
        ["", "CA", None, "Assessment, Authorization, and Monitoring", "", "PS", "Personnel Security"],
    ]
    text = _pdf_table_markdown(rows, rows[0], False)
    assert text.splitlines() == [
        "| ID | FAMILY | ID | FAMILY |",
        "| --- | --- | --- | --- |",
        "| AC | Access Control | PE | Physical and Environmental Protection |",
        "| CA | Assessment, Authorization, and Monitoring | PS | Personnel Security |",
    ]
    spanned = _pdf_table_markdown([["Limits", None], ["1", "2"]], [], False)
    assert spanned.splitlines()[0] == "| Limits | Limits |"
    assert _pdf_table_markdown([["", None]], [], False) == ""


def test_unknown_parser_version_is_rejected() -> None:
    try:
        DocumentParser(version="v3")  # type: ignore[arg-type]
    except ValueError as error:
        assert "v3" in str(error)
    else:
        raise AssertionError("expected ValueError")


HTML = """<h2 id="x" data-section="1">
  Authentication Assurance Levels
</h2>
<p><a name="t2">Table 2. Summary of secrets</a></p>
<table>
  <thead><tr><th colspan="2">Secret</th><th>Section</th></tr>
         <tr><th>Type</th><th>Purpose</th><th></th></tr></thead>
  <tbody>
    <tr><td rowspan="2">Look-up secret</td><td><p>Used once</p><p>to prove a|b</p></td><td>3.1.2</td></tr>
    <tr><td>Second purpose</td><td>3.1.3</td></tr>
  </tbody>
</table>
<p>After.</p>
"""


def test_markdown_extractor_keeps_headings_and_table_rows() -> None:
    extractor = MarkdownExtractor()
    extractor.feed(HTML)
    lines = [line for line in extractor.text().splitlines() if line]
    assert lines == [
        "## Authentication Assurance Levels",
        "Table 2. Summary of secrets",
        "| Secret / Type | Secret / Purpose | Section |",
        "| --- | --- | --- |",
        "| Look-up secret | Used once to prove a&#124;b | 3.1.2 |",
        "| Look-up secret | Second purpose | 3.1.3 |",
        "After.",
    ]
    parsed = v2().parse_text(extractor.text())
    [table] = tables(parsed)
    assert table.metadata["caption"] == "Table 2. Summary of secrets"
    assert table.section_path == ("Authentication Assurance Levels",)


def test_text_extractor_v1_unchanged() -> None:
    extractor = TextExtractor()
    extractor.feed(HTML)
    text = extractor.text()
    assert "## \n" in text and "## Authentication Assurance Levels" not in text
    assert "| Secret / Type" not in text


def test_each_dataset_names_the_parser_of_its_chunks() -> None:
    from pathlib import Path

    from eval_dataset import load_dataset
    from evaluate_suite import parser_version_of

    data = Path(__file__).resolve().parents[1] / "data"
    assert parser_version_of(load_dataset(data / "eval_dataset_v3.json")) == "v1"
    assert parser_version_of(load_dataset(data / "xquad_dataset_full.json")) == "v1"
    assert parser_version_of(load_dataset(data / "eval_dataset_v3_parser_v2.json")) == "v2"
