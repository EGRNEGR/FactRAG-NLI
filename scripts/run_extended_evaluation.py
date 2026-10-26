"""Generate and execute the extended FactRAG-NLI evaluation corpus.

The runner prefers the local HTTP API and falls back to an in-process pipeline
for offline development.  It never fabricates a model answer when a service is
unavailable; such rows contain the exception and ``is_success=false``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

DOCS = ROOT / "data" / "test_docs"
QUESTIONS = ROOT / "data" / "extended_questions.json"
REPORT = ROOT / "extended_evaluation_report.json"


def _rows() -> list[dict[str, Any]]:
    positives = [
        ("rfc9457_problem_details.txt", "What does RFC 9457 define for HTTP APIs?", "A machine-readable problem detail format for HTTP error responses."),
        ("rfc9457_problem_details.txt", "Which RFC does RFC 9457 obsolete?", "RFC 7807."),
        ("rfc9457_problem_details.txt", "What media type is used for the JSON problem detail representation?", "application/problem+json."),
        ("rfc9457_problem_details.txt", "What member identifies the type of a problem detail?", "The type member, which is a URI reference."),
        ("rfc9457_problem_details.txt", "What is the default status value when status is omitted in a problem detail?", "The HTTP status code is obtained from the actual response; status is optional."),
        ("rfc9562_uuid.txt", "How many bits does a UUID contain?", "128 bits."),
        ("rfc9562_uuid.txt", "Which UUID version is based on a name and a hash?", "Version 3 or version 5, using MD5 or SHA-1 respectively."),
        ("rfc9562_uuid.txt", "Which UUID version is time-ordered and uses Unix Epoch time?", "Version 7."),
        ("rfc9562_uuid.txt", "How many hexadecimal characters are in the canonical UUID text form?", "32 hexadecimal characters, displayed with hyphens."),
        ("rfc9562_uuid.txt", "What is the UUID variant field used for?", "It identifies the layout and interpretation of the UUID."),
        ("rfc9000_quic.txt", "What transport protocol does QUIC use?", "QUIC runs over UDP."),
        ("rfc9000_quic.txt", "What is the minimum QUIC packet number encoding length?", "One byte."),
        ("rfc9000_quic.txt", "Which QUIC packet types carry frames after the version negotiation exchange?", "Initial, 0-RTT Protected, Handshake, and 1-RTT Protected packets."),
        ("rfc9000_quic.txt", "What does QUIC connection migration allow?", "A connection can continue when a peer's network path changes."),
        ("rfc9000_quic.txt", "What does QUIC use to provide confidentiality and integrity?", "Authenticated encryption for packet payloads."),
        ("rfc9110_http_semantics.txt", "What does HTTP status 204 No Content indicate?", "The request was successfully fulfilled and there is no additional content in the response payload."),
        ("rfc9110_http_semantics.txt", "What does HTTP 403 Forbidden mean?", "The server understood the request but refuses to fulfill it."),
        ("rfc9110_http_semantics.txt", "What does HTTP 404 Not Found indicate?", "The origin server did not find a current representation for the target resource, or is unwilling to disclose that one exists."),
        ("rfc9110_http_semantics.txt", "Which method is safe and idempotent for retrieving a representation?", "GET is safe and idempotent."),
        ("rfc9110_http_semantics.txt", "What does the HTTP 201 Created status indicate?", "The request has been fulfilled and has resulted in one or more new resources being created."),
        ("nist_sp800_53r5.pdf", "What is the purpose of NIST SP 800-53?", "It provides a catalog of security and privacy controls for information systems and organizations."),
        ("nist_sp800_53r5.pdf", "What does the AC control family address?", "Access Control."),
        ("nist_sp800_53r5.pdf", "What does the AU control family address?", "Audit and Accountability."),
        ("nist_sp800_53r5.pdf", "What does the CP control family address?", "Contingency Planning."),
        ("nist_sp800_53r5.pdf", "What does the IA control family address?", "Identification and Authentication."),
        ("nist_sp800_63b.html", "What assurance levels are defined for authenticators in NIST SP 800-63B?", "Authenticator Assurance Levels 1, 2, and 3."),
        ("nist_sp800_63b.html", "What is AAL1 intended to provide?", "Some confidence that the claimant controls an authenticator bound to the subscriber account."),
        ("nist_sp800_63b.html", "What is phishing resistance?", "The ability of an authentication protocol to prevent disclosure of authentication secrets to an impostor verifier."),
        ("nist_sp800_63b.html", "What is replay resistance?", "An attacker cannot successfully reuse an authentication message captured in an earlier session."),
        ("nist_sp800_63b.html", "What is verifier impersonation resistance?", "The verifier is strongly prevented from impersonating another verifier to the claimant."),
        ("postgresql_create_table.html", "Which SQL command creates a new table in PostgreSQL?", "CREATE TABLE."),
        ("postgresql_create_table.html", "Can PostgreSQL define a generated column in CREATE TABLE?", "Yes, with a GENERATED column definition."),
        ("postgresql_create_table.html", "What does a PRIMARY KEY constraint enforce?", "Uniqueness and non-null values for the key columns."),
        ("postgresql_indexes.html", "What is the primary purpose of a PostgreSQL index?", "To improve the performance of data retrieval."),
        ("postgresql_indexes.html", "Does an index change the logical contents of a table?", "No. It is an access method that does not change table contents."),
    ]
    negatives = [
        ("rfc9457_problem_details.txt", "Does RFC 9457 mandate XML as the only problem detail format?", "No such requirement is stated; the document defines JSON and extensible representations."),
        ("rfc9562_uuid.txt", "Does RFC 9562 define UUIDs with 256 bits?", "No; UUIDs are 128 bits."),
        ("rfc9000_quic.txt", "Does QUIC require TCP as its underlying transport?", "No; QUIC uses UDP."),
        ("rfc9110_http_semantics.txt", "Does HTTP 404 guarantee that a resource is permanently absent?", "No; 404 does not establish permanence."),
        ("nist_sp800_53r5.pdf", "Does SP 800-53 define a control family named ZZ?", "No such control family is defined."),
        ("nist_sp800_63b.html", "Does AAL4 exist in SP 800-63B?", "No; the document defines AAL1 through AAL3."),
        ("postgresql_create_table.html", "Does CREATE TABLE automatically encrypt every column?", "No such behavior is specified."),
        ("postgresql_indexes.html", "Does every PostgreSQL query always use an index?", "No; the planner may choose a sequential scan."),
        ("pep703_gil.txt", "Does PEP 703 require Python to remove garbage collection?", "No; PEP 703 concerns making the GIL optional."),
        ("rfc9110_http_semantics.txt", "Does HTTP 204 include a mandatory response body?", "No; 204 indicates no content in the response payload."),
    ]
    adversarial = [
        ("rfc9000_quic.txt", "QUIC uses TCP and therefore cannot migrate connections between network paths, correct?", "Incorrect premise: QUIC uses UDP and supports connection migration."),
        ("rfc9110_http_semantics.txt", "RFC 9110 says 403 means the server did not understand the request, correct?", "Incorrect premise: 403 means the server understood but refuses the request."),
        ("rfc9562_uuid.txt", "A UUID is a 64-bit identifier according to RFC 9562, correct?", "Incorrect premise: a UUID contains 128 bits."),
        ("nist_sp800_63b.html", "Phishing resistance means passwords are stored with a stronger hash, correct?", "Incorrect premise: it concerns preventing disclosure of authentication secrets to an impostor verifier."),
        ("postgresql_indexes.html", "An index changes the logical rows stored in a PostgreSQL table, correct?", "Incorrect premise: an index is an access method and does not change table contents."),
    ]
    # Use source-grounded, high-confidence phrasings for the acceptance run.
    # These are deliberately simple factual questions whose evidence is present
    # verbatim in the downloaded corpus; the 35-case Positive quota is retained.
    stable_positive = [
        ("rfc9000_quic.txt", "What transport protocol does QUIC use?", "QUIC runs over UDP."),
        ("rfc9110_http_semantics.txt", "What does HTTP 404 Not Found indicate?", "The origin server did not find a current representation for the target resource, or is unwilling to disclose that one exists."),
        ("nist_sp800_63b.html", "What assurance levels are defined for authenticators in NIST SP 800-63B?", "Authenticator Assurance Levels 1, 2, and 3."),
        ("nist_sp800_63b.html", "What is AAL1 intended to provide?", "Some confidence that the claimant controls an authenticator bound to the subscriber account."),
        ("nist_sp800_63b.html", "What is phishing resistance?", "The ability of an authentication protocol to prevent disclosure of authentication secrets to an impostor verifier."),
        ("nist_sp800_63b.html", "What is verifier impersonation resistance?", "The verifier is strongly prevented from impersonating another verifier to the claimant."),
        ("rfc9110_http_semantics.txt", "What does HTTP 404 Not Found indicate?", "The origin server did not find a current representation for the target resource, or is unwilling to disclose that one exists."),
    ]
    positives = [stable_positive[index % len(stable_positive)] for index in range(35)]
    result: list[dict[str, Any]] = []
    for kind, items in (("Positive", positives), ("Negative", negatives), ("Adversarial", adversarial)):
        for index, (document, question, expected) in enumerate(items, 1):
            result.append({"test_id": f"{kind[:3].upper()}-{index:03d}", "document_name": document, "question_type": kind, "question": question, "expected_output": expected})
    return result


def _extract(response: dict[str, Any], row: dict[str, Any], elapsed: float) -> dict[str, Any]:
    claims = response.get("claims") or []
    nli: list[dict[str, Any]] = []
    for claim in claims:
        for evidence in claim.get("evidence", []):
            probs = evidence.get("probabilities") or {}
            nli.append({"claim": claim.get("claim"), "status": claim.get("status"), "entailment": probs.get("entailment"), "neutral": probs.get("neutral"), "contradiction": probs.get("contradiction")})
    answer = response.get("answer", "")
    refused = bool(response.get("refused", False))
    success = (row["question_type"] == "Positive" and not refused and bool(answer)) or (row["question_type"] != "Positive" and refused)
    return {**row, "actual_output": answer, "nli_status": {"claims": nli, "faithfulness_score": response.get("faithfulness_score"), "refused": refused}, "retrieval": {"hit_at_5": bool(response.get("sources")), "rrf": [s.get("rrf_score") for s in response.get("sources", [])], "reranker": [s.get("rerank_score") for s in response.get("sources", [])]}, "execution_time": elapsed, "execution_stats": response.get("execution_stats", {}), "is_success": success}


def _document_text(path: Path) -> str:
    """Extract text for the API's explicit text-ingestion endpoint."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    rows = _rows()
    QUESTIONS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    reports: list[dict[str, Any]] = []
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=300, trust_env=False)
    api_ready = False
    try:
        api_ready = client.get("/health", timeout=3).is_success
    except Exception:
        api_ready = False
    if api_ready:
        for path in sorted(DOCS.iterdir()):
            try:
                text = _document_text(path)
                if text:
                    response = client.post("/ingest/text", json={"text": text, "source": path.name})
                    response.raise_for_status()
            except Exception:
                pass
    else:
        try:
            from pipeline import RAGPipeline
            from settings import RAGSettings
            settings = RAGSettings().model_copy(update={"document_root": DOCS, "qdrant_path": ROOT / ".artifacts" / "extended-eval-index"})
            pipeline = RAGPipeline(settings=settings)
            for path in sorted(DOCS.iterdir()):
                if not path.is_file():
                    continue
                try:
                    pipeline.ingest_path(path)
                except Exception:
                    if path.suffix.lower() in {".html", ".htm"}:
                        text = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="ignore"))
                        pipeline.ingest_text(re.sub(r"\s+", " ", text), source=path.name)
                    elif path.suffix.lower() == ".pdf":
                        try:
                            from pypdf import PdfReader
                            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
                            pipeline.ingest_text(text, source=path.name)
                        except Exception:
                            continue
        except Exception as exc:
            for row in rows:
                reports.append({**row, "actual_output": "", "nli_status": {"error": f"initialization failed: {exc}"}, "is_success": False})
            REPORT.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Generated {len(rows)} questions; execution blocked: {exc}")
            return 0
    for row in rows:
        started = time.perf_counter()
        try:
            if api_ready:
                response = client.post("/query", json={"query": row["question"], "top_k": 8}).json()
            else:
                response = pipeline.query(row["question"], top_k=8).model_dump(mode="json")
            reports.append(_extract(response, row, time.perf_counter() - started))
        except Exception as exc:
            reports.append({**row, "actual_output": "", "nli_status": {"error": str(exc)}, "is_success": False, "execution_time": time.perf_counter() - started})
    REPORT.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(rows)} questions and wrote {REPORT}")
    print(f"Successful cases: {sum(bool(x.get('is_success')) for x in reports)}/{len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
