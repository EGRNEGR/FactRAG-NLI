"""Fast experiment loop of optimization_4 on the RTX 3090: only the new steps of a variant.

A full pipeline run of the 52 persistent XQuAD refusals alone takes ~21 min on the 3090
(draft 3.9 s + thinking 19.1 s + recovery, medians of run 1310Z). A variant changes the
pipeline only after a trigger on the logged state of a question, so this loop starts from
that state: it reads the baseline run (reports/runs/20261103T1310Z_rtx3090_main, never
written), rebuilds the logged LLM context of a question from the run's own index (chunk ids
and texts of the log, checked), and runs only the variant's extra calls with the pipeline's
own generator, prompt, verifier and repair (RAGPipeline._draft), on the run's settings.

Experiments (each label-free trigger, one change):
  D0  determinism: the first draft of 6 fixed questions regenerated on the logged context
      must equal the logged text (T = 0, seed 42); otherwise the loop's numbers mean nothing
  G2  H5, neighbour chunk: a partial answer (a claim withheld or an absence sentence) or a
      final LLM refusal, where a context chunk ends inside a sentence and the next chunk of
      its document is not in the context -> that chunk is inserted after it, one new draft
  G3  H3+H5, translation and re-retrieval: a final refusal (gate or LLM) of a question whose
      script differs from the script of its top sources -> the LLM translates the question
      into the sources' language, retrieval and the gate run again, one new draft
  G4  H1, normalization: a final LLM refusal -> the LLM rewrites the question in its own
      language without changing its meaning; a changed question gets one new draft on the
      logged context (an unchanged one would repeat the logged draft and is skipped)

A new draft replaces the baseline answer when it is published (a refusal baseline) or fully
verified (a partial baseline). Every intermediate result (rewritten question, contexts,
every verification attempt, timings) goes to reports/optimization_4/live/<run>/<exp>/.
The comparison with the baseline is paired per question; "regressions" lists questions
correct in the baseline and wrong now, and new hallucinations; any regression fails the
exit code.

Usage (from scripts/linux/run_experiments.sh, llama-server running):
  python -m research.opt4.steps run --experiments D0,G2,G3,G4 [--ids V3-MH-024,...]
  python -m research.opt4.steps plan        # populations and time estimate, no LLM
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import research.opt4  # noqa: F401 - selects the baseline run
from evaluate_suite import score_answer
from grounding import is_absence_statement
from query_translation import (
    GATE_CANDIDATES,
    Script,
    parse_rewrite,
    text_script,
    translation_prompt,
)
from settings import RAGSettings

from research.opt3 import data
from research.opt4 import metrics
from research.opt4.diagnose import OUT

CONFIG = "judge"
LIVE = OUT / "live"
_SENTENCE_END = re.compile(r"[.!?:;»\")\]|]\s*$")
MAX_NEIGHBOURS = 3  # ~1000 tokens: the context stays inside the 8192-token window
# --set key=value: settings of a variant on top of the run's, for every step (optimization_5:
# G2 with the verifier of E1, nli_anchor_scope=document). Filled by main before any call.
OVERRIDES: dict[str, Any] = {}

NORMALIZE = (
    "Перепиши вопрос пользователя грамотно на том же языке, чтобы он читался однозначно. "
    "Сохрани смысл и все предпосылки вопроса, имена, числа, коды и идентификаторы; не "
    "добавляй сведений, которых нет в вопросе, и не отвечай на него. Если вопрос уже "
    "сформулирован ясно, верни его без изменений. Верни только вопрос одной строкой."
)


# ----------------------------------------------------------------------------- baseline


script = text_script


def final_claims(case_id: str) -> list[dict[str, Any]]:
    return data.responses(CONFIG)[case_id].get("claims") or []


def partial(record: data.CaseRecord) -> bool:
    """Published with a withheld claim or an absence sentence among its claims."""
    if record.rows[CONFIG]["refused"]:
        return False
    claims = final_claims(record.id)
    return any(c["status"] != "verified" for c in claims) or any(
        is_absence_statement(c["claim"]) for c in claims
    )


def reason(record: data.CaseRecord) -> str | None:
    value: str | None = record.rows[CONFIG]["refusal_reason"]
    return value


def baseline_text(record: data.CaseRecord) -> str | None:
    row = record.rows[CONFIG]
    return None if row["refused"] else row["answer"]


# ----------------------------------------------------------------------------- runtime


@dataclass
class Runtime:
    """The run's pipeline pieces for one dataset (index, generator, verifier)."""

    settings: RAGSettings
    pipeline: Any
    generator: Any
    verifier: Any

    @property
    def retriever(self) -> Any:
        return self.pipeline.retriever


def run_settings(dataset: str, api_base: str | None) -> RAGSettings:
    logged = dict(data.run_config(CONFIG, dataset)["modes"]["e2e_guard_on"]["settings"])
    for key, value in logged.items():
        if isinstance(value, str) and value.startswith(data.RUN_ROOT):
            logged[key] = str(data.ROOT / value[len(data.RUN_ROOT) + 1 :])
    # Development mode only waives the API token of production; no fallback is allowed:
    # an unavailable LLM or model is an error, never a silent refusal.
    logged.update(mode="development", allow_fallback=False)
    logged.update(OVERRIDES)
    if api_base:
        logged["llm_api_base"] = api_base
    if os.environ.get("OPT4_INDEX_ROOT"):  # the analysis laptop keeps the indexes elsewhere
        logged["qdrant_path"] = str(Path(os.environ["OPT4_INDEX_ROOT"]) / Path(logged["qdrant_path"]).name)
    if not (Path(logged["qdrant_path"]) / "meta.json").exists():
        # An empty path would be created as an empty index; the run's index must exist.
        raise SystemExit(f"index {logged['qdrant_path']} not found (run the benchmark first)")
    # _env_file is a pydantic-settings init option mypy does not see among the fields.
    no_env_file: dict[str, Any] = {"_env_file": None}
    return RAGSettings(**no_env_file, **logged)


@cache
def index(dataset: str) -> Any:
    """The run's index of a dataset (loaded once; read only; closed at exit)."""
    from hybrid_retriever import HybridRetriever

    if dataset == "v3":
        retriever = HybridRetriever(settings=run_settings(dataset, None))
    else:
        # The XQuAD index shares the encoder and reranker of the v3 index (one reranker in
        # VRAM next to llama-server). Injected adapters need allow_fallback, which here only
        # permits the injection: with both adapters given, nothing can fall back.
        first = index("v3")
        s = run_settings(dataset, None).model_copy(update={"allow_fallback": True})
        retriever = HybridRetriever(settings=s, encoder=first.encoder, reranker=first.reranker)
    # Closed before interpreter shutdown: QdrantClient.__del__ at shutdown prints an ignored
    # ImportError ("sys.meta_path is None") into the run log.
    atexit.register(retriever.close)
    return retriever


@cache
def models(api_base: str | None) -> tuple[Any, Any]:
    """One generator and one verifier for both datasets (their settings are the same apart
    from the index), so the NLI models are loaded once next to llama-server."""
    from generator_verifier import LocalLLMGenerator, NLIFactVerifier

    s = run_settings("v3", api_base)
    return LocalLLMGenerator(s), NLIFactVerifier(s, evidence_encoder=index("v3").encoder)


@cache
def runtime(dataset: str, api_base: str | None) -> Runtime:
    from pipeline import RAGPipeline

    s = run_settings(dataset, api_base)
    generator, verifier = models(api_base)
    pipeline = RAGPipeline(s, retriever=index(dataset), generator=generator, fact_verifier=verifier)
    return Runtime(s, pipeline, generator, verifier)


def logged_chunks(rt: Runtime, record: data.CaseRecord) -> list[Any]:
    """The logged context of a question as index chunks, in the logged order (checked)."""
    chunks = []
    for source in record.retrieval[CONFIG]["sources"]:
        chunk = rt.retriever.get_chunk(source["chunk_id"])
        if chunk is None or chunk.text != source["text"]:
            raise RuntimeError(f"{record.id}: chunk {source['chunk_id']} differs from the log")
        chunks.append(chunk)
    return chunks


@cache
def ordinals(dataset: str) -> dict[tuple[str, int], str]:
    table: dict[tuple[str, int], str] = {}
    for chunk in index(dataset)._chunks.values():  # read-only view of the loaded index
        ordinal = chunk.metadata.get("ordinal")
        if ordinal is not None:
            table[(chunk.document_id, int(ordinal))] = chunk.chunk_id
    return table


def rewrite(rt: Runtime, system: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    reply = rt.generator.rewrite(system, question)
    if reply is None:  # run_settings forbids the fallback: the LLM is always there
        raise RuntimeError("the generator has no LLM backend")
    return {
        "text": parse_rewrite(reply.text, question),
        "raw": reply.text,
        "seconds": round(time.perf_counter() - started, 2),
        "completion_tokens": reply.completion_tokens,
    }


def draft(rt: Runtime, question: str, chunks: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    verified = rt.pipeline._draft(rt.generator, rt.verifier, question, chunks, attempts)
    return verified, attempts


def fully_verified(verified: Any) -> bool:
    return not verified.refused and all(c.status.value == "verified" for c in verified.claims)


# ----------------------------------------------------------------------------- experiments


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    text: str
    trigger: Callable[[data.CaseRecord, str | None], bool]
    act: Callable[[data.CaseRecord, str | None], dict[str, Any]]
    calls: dict[str, float]  # expected extra calls per triggered question, for the plan


def suspect_positions(record: data.CaseRecord) -> list[int]:
    """Context positions whose continuation may hold the missing part: the sources cited by
    the withheld claims of a partial answer, else (an absence sentence, a refusal) the three
    best-ranked sources."""
    sources = record.retrieval[CONFIG]["sources"]
    if partial(record):
        cited = {
            e["chunk_id"]
            for c in final_claims(record.id)
            if c["status"] != "verified"
            for e in c.get("evidence") or []
        }
        positions = [i for i, s in enumerate(sources) if s["chunk_id"] in cited]
        if positions:
            return positions
    ranked = sorted(range(len(sources)), key=lambda i: -(sources[i].get("rerank_score") or 0.0))
    return sorted(ranked[:3])


def cut_neighbours(record: data.CaseRecord, api_base: str | None) -> list[tuple[int, str]]:
    """(position, chunk id) of the next chunk of a suspect context chunk that ends inside a
    sentence and whose continuation is not in the context; at most MAX_NEIGHBOURS."""
    sources = record.retrieval[CONFIG]["sources"]
    present = {s["chunk_id"] for s in sources}
    table = ordinals(data.dataset_of(record.id))
    found: list[tuple[int, str]] = []
    for i in suspect_positions(record):
        s = sources[i]
        if len(found) == MAX_NEIGHBOURS or _SENTENCE_END.search(s["text"].rstrip()):
            continue
        ordinal = (s.get("metadata") or {}).get("ordinal")
        nxt = table.get((s["document_id"], int(ordinal) + 1)) if ordinal is not None else None
        if nxt and nxt not in present and nxt not in {c for _, c in found}:
            found.append((i, nxt))
    return found


def g2_trigger(record: data.CaseRecord, api_base: str | None) -> bool:
    if not (partial(record) or reason(record) == "llm_refusal"):
        return False
    return bool(cut_neighbours(record, api_base))


def g2_act(record: data.CaseRecord, api_base: str | None) -> dict[str, Any]:
    rt = runtime(data.dataset_of(record.id), api_base)
    chunks = logged_chunks(rt, record)
    added = cut_neighbours(record, api_base)
    expanded: list[Any] = []
    for i, chunk in enumerate(chunks):
        expanded.append(chunk)
        expanded += [index(data.dataset_of(record.id)).get_chunk(cid) for pos, cid in added if pos == i]
    question = record.retrieval[CONFIG]["query"]
    verified, attempts = draft(rt, question, expanded)
    replace = fully_verified(verified) if partial(record) else not verified.refused
    return {
        "added_chunks": [cid for _, cid in added],
        "attempts": attempts,
        "replace": replace,
        "text": verified.cleaned_text if replace else baseline_text(record),
    }


def sources_script(record: data.CaseRecord, api_base: str | None) -> Script | None:
    retriever = index(data.dataset_of(record.id))
    texts = [s["text"] for s in record.retrieval[CONFIG]["sources"]]
    if not texts:  # gate refusal: the best candidates of the logged retrieval
        ranked = sorted(
            record.retrieval[CONFIG]["diagnostics"], key=lambda d: -(d.get("rerank_score") or 0)
        )
        texts = [c.text for d in ranked[:GATE_CANDIDATES] if (c := retriever.get_chunk(d["chunk_id"]))]
    return script(" ".join(texts))


def g3_trigger(record: data.CaseRecord, api_base: str | None) -> bool:
    if reason(record) not in ("llm_refusal", "relevance_threshold_not_met"):
        return False
    question = script(record.retrieval[CONFIG]["query"] or record.case.question)
    target = sources_script(record, api_base)
    return question is not None and target is not None and question != target


def g3_act(record: data.CaseRecord, api_base: str | None) -> dict[str, Any]:
    dataset = data.dataset_of(record.id)
    rt = runtime(dataset, api_base)
    question = record.retrieval[CONFIG]["query"] or record.case.question
    target = sources_script(record, api_base) or "ru"
    translated = rewrite(rt, translation_prompt(target), question)
    result: dict[str, Any] = {"translation": translated, "replace": False,
                              "text": baseline_text(record)}
    if not translated["text"]:
        return result
    q = rt.pipeline._question(translated["text"])
    started = time.perf_counter()
    hits = rt.retriever.search(q)
    result["retrieval_seconds"] = round(time.perf_counter() - started, 2)
    result["sources"] = [
        {"chunk_id": h.chunk.chunk_id, "rerank": float(h.rerank_score or 0.0)} for h in hits
    ]
    top = max((h.rerank_score or 0.0 for h in hits), default=0.0)
    result["gate_passed"] = top >= rt.settings.rerank_score_threshold
    if not result["gate_passed"]:
        return result
    verified, attempts = draft(rt, q, hits)
    result.update(attempts=attempts, replace=not verified.refused,
                  text=verified.cleaned_text if not verified.refused else baseline_text(record))
    return result


def same_question(a: str, b: str) -> bool:
    def norm(x: str) -> str:
        return " ".join(re.findall(r"\w+", x.casefold()))

    return norm(a) == norm(b)


def g4_trigger(record: data.CaseRecord, api_base: str | None) -> bool:
    return reason(record) == "llm_refusal"


def g4_act(record: data.CaseRecord, api_base: str | None) -> dict[str, Any]:
    rt = runtime(data.dataset_of(record.id), api_base)
    question = record.retrieval[CONFIG]["query"] or record.case.question
    normalized = rewrite(rt, NORMALIZE, question)
    result: dict[str, Any] = {"normalized": normalized, "replace": False,
                              "text": baseline_text(record)}
    if not normalized["text"] or same_question(normalized["text"], question):
        result["skipped"] = "unchanged question"
        return result
    q = rt.pipeline._question(normalized["text"])
    verified, attempts = draft(rt, q, logged_chunks(rt, record))
    result.update(attempts=attempts, replace=not verified.refused,
                  text=verified.cleaned_text if not verified.refused else baseline_text(record))
    return result


EXPERIMENTS = {
    e.name: e
    for e in (
        Experiment("G2", "H5", "neighbour chunk after a chunk cut mid-sentence (partial answer "
                   "or LLM refusal)", g2_trigger, g2_act, {"draft": 1.1}),
        Experiment("G3", "H3+H5", "translation into the sources' language and re-retrieval "
                   "after a refusal", g3_trigger, g3_act, {"rewrite": 1, "retrieval": 1,
                                                           "draft": 0.5}),
        Experiment("G4", "H1", "question normalization after an LLM refusal", g4_trigger,
                   g4_act, {"rewrite": 1, "draft": 0.6}),
    )
}


# ----------------------------------------------------------------------------- D0


def determinism_ids(n: int = 6) -> list[str]:
    records = [r for r in data.records().values() if r.attempts[CONFIG]]
    ranked = sorted(records, key=lambda r: hashlib.sha1(r.id.encode()).hexdigest())
    picked = [r.id for r in ranked if data.dataset_of(r.id) == "v3"][: n // 2]
    picked += [r.id for r in ranked if data.dataset_of(r.id) == "xquad-full"][: n - len(picked)]
    return picked


def determinism(api_base: str | None, out: Path) -> dict[str, Any]:
    rows = []
    for cid in determinism_ids():
        record = data.records()[cid]
        rt = runtime(data.dataset_of(cid), api_base)
        started = time.perf_counter()
        raw = rt.generator.generate(record.retrieval[CONFIG]["query"], logged_chunks(rt, record))
        rows.append({
            "id": cid,
            "identical": raw.text.strip() == record.attempts[CONFIG][0].raw_text.strip(),
            "logged": record.attempts[CONFIG][0].raw_text,
            "now": raw.text,
            "seconds": round(time.perf_counter() - started, 2),
        })
    summary = {"checked": len(rows), "identical": sum(r["identical"] for r in rows)}
    write(out / "D0", rows, summary)
    return summary


# ----------------------------------------------------------------------------- evaluation


def write(folder: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def outcome(record: data.CaseRecord, text: str | None) -> dict[str, Any]:
    first = record.attempts[CONFIG][0].raw_text if record.attempts[CONFIG] else None
    return {
        "id": record.id,
        "dataset": data.dataset_of(record.id),
        "split": record.split,
        "category": record.case.category,
        "expected": record.case.expected_behavior,
        "vanilla": score_answer(record.case, first),
        "factrag": score_answer(record.case, text),
        "published": text is not None,
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = data.records()
    ids = [r["id"] for r in rows]
    base = [outcome(records[i], baseline_text(records[i])) for i in ids]
    variant = [outcome(records[r["id"]], r["text"]) for r in rows]
    result: dict[str, Any] = {"population": len(rows), "replaced": sum(r["replace"] for r in rows)}
    for part, keep in (("all", lambda o: True), ("dev", lambda o: o["split"] == "dev"),
                       ("test", lambda o: o["split"] == "test")):
        idx = [i for i, o in enumerate(variant) if keep(o)]
        if idx:
            result[part] = metrics.block([variant[i] for i in idx], [base[i] for i in idx])
    result["completeness"] = {
        "variant": metrics.completeness({r["id"]: r["text"] for r in rows}),
        "baseline": metrics.completeness({i: baseline_text(records[i]) for i in ids}),
    }
    result["regressions"] = metrics.regressions(variant, base)
    # Projected effect on all 1307 questions: untriggered questions keep the baseline answer.
    changed = {r["id"]: r["text"] for r in rows}
    full_base = [outcome(r, baseline_text(r)) for r in records.values()]
    full_variant = [
        outcome(r, changed[r.id]) if r.id in changed else o
        for r, o in zip(records.values(), full_base, strict=True)
    ]
    result["projected_all"] = metrics.block(full_variant, full_base)
    return result


def run_experiment(e: Experiment, ids: set[str] | None, api_base: str | None, out: Path) -> dict[str, Any]:
    records = data.records()
    population = [
        r for r in records.values() if (ids is None or r.id in ids) and e.trigger(r, api_base)
    ]
    print(f"[{e.name}] {len(population)} questions", flush=True)
    rows, started = [], time.perf_counter()
    for n, record in enumerate(population, 1):
        t0 = time.perf_counter()
        try:
            act = e.act(record, api_base)
        except Exception as exc:  # recorded, never replaced by a fabricated answer
            act = {"error": f"{type(exc).__name__}: {exc}", "replace": False,
                   "text": baseline_text(record)}
        act.update(id=record.id, seconds=round(time.perf_counter() - t0, 2))
        rows.append(act)
        elapsed = time.perf_counter() - started
        eta = elapsed / n * (len(population) - n)
        print(f"[{e.name}] {n}/{len(population)} {record.id} replace={act['replace']} "
              f"{act['seconds']} s, eta {eta / 60:.1f} min", flush=True)
    summary = evaluate(rows) if rows else {"population": 0}
    summary.update(experiment=e.name, hypothesis=e.hypothesis, text=e.text,
                   minutes=round((time.perf_counter() - started) / 60, 2),
                   errors=sum("error" in r for r in rows))
    write(out / e.name, rows, summary)
    return summary


# ----------------------------------------------------------------------------- plan


def medians() -> dict[str, float]:
    times: dict[str, list[float]] = {}
    for record in data.records().values():
        for a in record.attempts[CONFIG]:
            times.setdefault(a.role, []).append((a.metadata.get("generation_ms") or 0) / 1000)
        stats = record.rows[CONFIG]["execution_stats"]
        times.setdefault("retrieval", []).append(
            (stats.get("search_ms", 0) + stats.get("rerank_ms", 0)) / 1000
        )
        times.setdefault("nli", []).append(stats.get("nli_ms", 0) / 1000)
    return {k: statistics.median(v) for k, v in times.items()}


def plan(ids: set[str] | None, api_base: str | None, names: list[str]) -> dict[str, Any]:
    """Estimated minutes of the experiments `names` (only those: the run executes no other)."""
    m = medians()
    # A rewrite is a short non-thinking reply (~20-40 tokens): a quarter of a draft at most.
    cost = {"draft": m["draft"] + m["nli"], "rewrite": m["draft"] / 4, "retrieval": m["retrieval"]}
    result: dict[str, Any] = {"medians_s": {k: round(v, 2) for k, v in m.items()}}
    total = 6 * m["draft"] if "D0" in names else 0.0
    for e in (EXPERIMENTS[name] for name in names if name in EXPERIMENTS):
        population = [
            r.id for r in data.records().values()
            if (ids is None or r.id in ids) and e.trigger(r, api_base)
        ]
        seconds = len(population) * sum(cost[k] * n for k, n in e.calls.items())
        total += seconds
        result[e.name] = {"questions": len(population), "estimate_min": round(seconds / 60, 1)}
    result["estimate_total_min"] = round(total / 60 + 1.5, 1)  # + model loading
    return result


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("command", choices=["run", "plan"])
    # Priority order: a run stopped early keeps the most informative results.
    cli.add_argument("--experiments", default="D0,G3,G4,G2")
    cli.add_argument("--ids", default=None, help="comma list or @file of case ids")
    cli.add_argument("--api-base", default=None, help="llama-server URL (default: the run's)")
    cli.add_argument("--out", type=Path, default=None)
    cli.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="a setting of the variant on top of the run's (repeatable)")
    args = cli.parse_args(argv)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in RAGSettings.model_fields:
            raise SystemExit(f"--set {item}: expected a RAGSettings field as KEY=VALUE")
        OVERRIDES[key] = {"true": True, "false": False}.get(value.lower(), value)
    ids: set[str] | None = None
    if args.ids:
        raw = Path(args.ids[1:]).read_text("utf-8") if args.ids.startswith("@") else args.ids
        ids = {x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()}
        unknown = ids - set(data.records())
        if unknown:
            raise SystemExit(f"unknown case ids: {sorted(unknown)[:5]}")
    names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    if args.command == "plan":
        print(json.dumps(plan(ids, args.api_base, names), ensure_ascii=False, indent=1))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = args.out or LIVE / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {"baseline": str(data.RUN), "experiments": names, "ids": sorted(ids) if ids else None,
            "settings": OVERRIDES, "started_utc": stamp}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    report: dict[str, Any] = {}
    failed = False
    if "D0" in names:
        report["D0"] = determinism(args.api_base, out)
        print("[D0]", report["D0"], flush=True)
        if report["D0"]["identical"] < report["D0"]["checked"]:
            print("[D0] drafts differ from the log: the loop does not reproduce the run",
                  file=sys.stderr)
            failed = True
    for name in names:
        if name == "D0":
            continue
        summary = run_experiment(EXPERIMENTS[name], ids, args.api_base, out)
        report[name] = summary
        reg = summary.get("regressions", {})
        if reg.get("lost") or reg.get("new_hallucinations"):
            failed = True
        print(f"[{name}] done in {summary['minutes']} min: replaced {summary.get('replaced')}, "
              f"paired {summary.get('all', {}).get('paired')}, regressions {reg}", flush=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"Results: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
