"""Validated, disk-only configuration for an offline RAG deployment."""

from __future__ import annotations

import json
import os
import secrets
import struct
import ipaddress
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty local artifact: {path}")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Git LFS pointer instead of artifact: {path}")


def validate_transformer_directory(path: Path) -> None:
    """Check config, tokenizer and all declared weight shards without loading a model.

    This checks completeness, not cryptographic integrity or model compatibility.
    Actual loaders must still reject corrupt tensors and incompatible architectures.
    """
    if not path.is_dir():
        raise ValueError(f"Local model directory does not exist: {path}")
    config = path / "config.json"
    _nonempty_file(config)
    if not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
        raise ValueError(f"Model config must be a JSON object: {config}")
    _nonempty_file(path / "tokenizer_config.json")
    tokenizers = [
        path / name
        for name in (
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "spm.model",
            "tokenizer.model",
        )
    ]
    if not any(item.is_file() for item in tokenizers):
        raise ValueError(f"No local tokenizer artifact in {path}")
    for item in tokenizers:
        if item.exists():
            _nonempty_file(item)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if not index.exists():
            continue
        value = json.loads(index.read_text(encoding="utf-8"))
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid weight map: {index}")
        for shard in mapping.values():
            if not isinstance(shard, str):
                raise ValueError(f"Non-string shard in {index}")
            target = (path / shard).resolve()
            if not target.is_relative_to(path.resolve()):
                raise ValueError(f"Weight shard escapes model directory: {shard}")
            _nonempty_file(target)
        return
    for name in ("model.safetensors", "pytorch_model.bin"):
        weight = path / name
        if weight.exists():
            _nonempty_file(weight)
            return
    raise ValueError(f"No supported local weights in {path}")


class RAGSettings(BaseSettings):
    """Load RAG_* environment variables; validate production artifacts eagerly.

    API tokens use a JSON array, for example RAG_API_TOKENS='["secret..."]'.
    Construction has no network or filesystem-write side effects.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    mode: Literal["production", "development"] = Field(
        default="production",
        validation_alias=AliasChoices("RAG_MODE", "RAG_ENVIRONMENT", "environment", "mode"),
    )
    allow_fallback: bool = False
    embedding_model_path: Path = Path("models/bge-m3")
    reranker_model_path: Path = Path("models/bge-reranker-large")
    nli_model_path: Path = Path("models/mdeberta-v3-nli")
    llm_model_path: Path = Path("models/llm.gguf")
    llm_backend: Literal["llama_cpp", "vllm"] = "llama_cpp"
    max_question_chars: int = Field(default=4000, ge=1, le=32000)
    llm_api_base: str | None = Field(
        default=None, description="Base URL of the local vLLM / llama-server API"
    )
    llm_api_model: str | None = None
    llm_api_timeout: float = Field(default=120.0, gt=0, le=3600, allow_inf_nan=False)
    llm_api_stream: bool = False
    llm_tokenizer_mode: Literal["bytes", "llama_server"] = "bytes"
    generation_language: Literal["ru", "source"] = "ru"
    generation_repair_attempts: int = Field(default=0, ge=0, le=2)
    nli_evidence_focus: bool = False

    @field_validator("llm_api_base", mode="before")
    @classmethod
    def local_api_url(cls, value: object) -> str | None:
        """Accept loopback HTTP endpoints only; blank disables HTTP mode."""
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("LLM API URL must be a string")
        parsed = urlsplit(value)
        host = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LLM API URL requires HTTP(S), without credentials/query/fragment")
        if host == "localhost":
            host = "127.0.0.1"
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("LLM API must use a loopback address")
        authority = f"[{host}]" if ":" in host else host
        if parsed.port is not None:
            authority += f":{parsed.port}"
        return urlunsplit((parsed.scheme, authority, parsed.path.rstrip("/"), "", ""))

    llm_context_window: int = Field(default=8192, ge=512, le=131072)
    llm_seed: int = Field(default=42, ge=0)
    llm_gpu_layers: int = Field(default=-1, ge=-1)
    llm_cpu_retry: bool = True
    nli_max_length: int = Field(default=512, ge=32, le=8192)
    document_root: Path = Path("data/documents")
    qdrant_path: Path = Field(
        default=Path("data/qdrant"),
        validation_alias=AliasChoices(
            "RAG_QDRANT_PATH", "RAG_QDRANT_STORAGE_PATH", "qdrant_storage_path", "qdrant_path"
        ),
    )
    qdrant_collection: str = Field(default="rag_chunks", pattern=r"^[A-Za-z0-9_-]{1,80}$")
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    rerank_batch_size: int = Field(default=8, ge=1, le=256)
    rerank_max_length: int = Field(default=512, ge=32, le=8192)
    rerank_candidates: int = Field(default=60, ge=10, le=200)
    model_device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(?::\d+)?)$")
    model_precision: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    fallback_dimension: int = Field(default=384, ge=32)
    network_policy: Literal["local_only", "air_gap"] = "local_only"
    network_probe_timeout: float = Field(default=3.0, gt=0, le=30)
    minimum_free_ram_gb: float = Field(default=16.0, ge=0, allow_inf_nan=False)
    minimum_free_disk_gb: float = Field(default=10.0, ge=0, allow_inf_nan=False)

    @property
    def environment(self) -> Literal["production", "development"]:
        """Compatibility name for the deployment mode."""
        return self.mode

    @property
    def qdrant_storage_path(self) -> Path:
        """Compatibility name for embedded Qdrant storage."""
        return self.qdrant_path

    api_tokens: tuple[SecretStr, ...] = Field(default=(), repr=False)
    max_upload_file_size_mb: int = Field(default=100, ge=1, le=100)
    max_document_pages: int = Field(default=500, ge=1, le=500)
    dense_candidates: int = Field(default=60, ge=30, le=100)
    sparse_candidates: int = Field(default=60, ge=30, le=100)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    rerank_top_k: int = Field(default=5, ge=5, le=10)
    # Threshold applies to sigmoid-normalized cross-encoder logits.
    rerank_score_threshold: float = Field(default=0.5, ge=0, le=1)
    rerank_rrf_alpha: float = Field(default=0.0, ge=0, le=1)
    rerank_window_tokens: int = Field(default=384, ge=32, le=448)
    nli_entailment_threshold: float = Field(default=0.4, ge=0, le=1)
    nli_contradiction_threshold: float = Field(default=0.2, ge=0, le=1)
    chunk_target_tokens: int = Field(default=420, ge=300, le=600)
    chunk_max_tokens: int = Field(default=600, ge=300, le=600)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    ingestion_target_tokens: int = Field(default=650, ge=500, le=800)
    ingestion_max_tokens: int = Field(default=800, ge=500, le=800)
    ingestion_overlap_tokens: int = Field(default=75, ge=50, le=100)
    ingestion_batch_size: int = Field(default=2, ge=1, le=2)
    ingestion_max_chunks: int = Field(default=10000, ge=1, le=100000)
    summary_pool_size: int = Field(default=30, ge=10, le=100)
    summary_top_k: int = Field(default=12, ge=10, le=15)
    summary_mmr_lambda: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    summary_contradiction_threshold: float = Field(default=0.3, ge=0, le=1)
    summary_similarity_threshold: float = Field(default=0.85, ge=0, le=1)
    summary_output_tokens: int = Field(default=768, ge=128, le=1024)
    summary_max_generation_calls: int = Field(default=16, ge=1, le=32)
    temperature: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        validation_alias=AliasChoices(
            "RAG_LLM_TEMPERATURE", "RAG_TEMPERATURE", "llm_temperature", "temperature"
        ),
    )
    top_p: float = Field(
        default=0.9,
        gt=0,
        le=1,
        validation_alias=AliasChoices("RAG_LLM_TOP_P", "RAG_TOP_P", "llm_top_p", "top_p"),
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=1024,
        validation_alias=AliasChoices(
            "RAG_LLM_MAX_TOKENS", "RAG_MAX_TOKENS", "llm_max_tokens", "max_tokens"
        ),
    )

    @property
    def llm_temperature(self) -> float:
        """Generation-specific compatibility name."""
        return self.temperature

    @property
    def llm_top_p(self) -> float:
        """Generation-specific compatibility name."""
        return self.top_p

    @property
    def llm_max_tokens(self) -> int:
        """Generation-specific compatibility name."""
        return self.max_tokens

    @field_validator(
        "embedding_model_path",
        "reranker_model_path",
        "nli_model_path",
        "llm_model_path",
        "document_root",
        "qdrant_path",
    )
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        """Resolve symlinks and relative paths against the startup directory."""
        resolved = value.expanduser().resolve()
        if str(resolved).startswith(("\\\\", "//")):
            raise ValueError("Network paths are not allowed in a local deployment")
        return resolved

    @model_validator(mode="after")
    def validate_deployment(self) -> Self:
        """Reject unsafe mode combinations and incomplete production assets."""
        if self.allow_fallback and self.mode != "development":
            raise ValueError("RAG_ALLOW_FALLBACK=true requires RAG_MODE=development")
        if self.summary_top_k > self.summary_pool_size:
            raise ValueError("summary_top_k must not exceed summary_pool_size")
        if self.llm_backend == "vllm" and self.llm_api_base is None:
            raise ValueError("llm_backend=vllm requires RAG_LLM_API_BASE")
        if not self.chunk_overlap_tokens < self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("Require overlap < target <= maximum chunk tokens")
        if (
            not self.ingestion_overlap_tokens
            < self.ingestion_target_tokens
            <= self.ingestion_max_tokens
        ):
            raise ValueError("Require ingestion overlap < target <= maximum chunk tokens")
        if self.llm_max_tokens + 128 >= self.llm_context_window:
            raise ValueError("LLM context must exceed output budget plus 128 template tokens")
        for path in (self.document_root, self.qdrant_path):
            if path.exists() and not path.is_dir():
                raise ValueError(f"Expected directory: {path}")
        values = [token.get_secret_value() for token in self.api_tokens]
        if len(set(values)) != len(values) or any(
            len(token) < 32 or token != token.strip() for token in values
        ):
            raise ValueError(
                "API tokens must be unique, at least 32 characters, without edge spaces"
            )
        if self.mode == "production" and not values:
            raise ValueError("At least one API token is required in production")
        if not self.allow_fallback:
            for path in (self.embedding_model_path, self.reranker_model_path, self.nli_model_path):
                validate_transformer_directory(path)
            if self.llm_model_path.is_dir():
                validate_transformer_directory(self.llm_model_path)
            else:
                _nonempty_file(self.llm_model_path)
                with self.llm_model_path.open("rb") as stream:
                    header = stream.read(24)
                if len(header) != 24 or header[:4] != b"GGUF":
                    raise ValueError("LLM artifact is not a GGUF file")
                version, tensors, _ = struct.unpack("<IQQ", header[4:])
                if version not in {2, 3} or tensors == 0:
                    raise ValueError("Unsupported or empty GGUF model")
        return self

    def resolve_document(self, requested: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve a document beneath the trusted root, including symlink targets."""
        path = (self.document_root / requested).resolve(strict=must_exist)
        if path == self.document_root or not path.is_relative_to(self.document_root):
            raise ValueError("Document path escapes document_root")
        if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            raise ValueError("Unsupported document extension")
        if must_exist:
            if not path.is_file():
                raise ValueError("Document must be a regular file")
            if not 0 < path.stat().st_size <= self.max_upload_file_size_mb * 1024 * 1024:
                raise ValueError("Document is empty or exceeds the upload limit")
        return path

    def accepts_api_token(self, candidate: str) -> bool:
        """Compare every configured token without an early exit."""
        accepted = False
        for token in self.api_tokens:
            accepted |= secrets.compare_digest(
                candidate.encode("utf-8"), token.get_secret_value().encode("utf-8")
            )
        return accepted

    @staticmethod
    def activate_offline_mode() -> None:
        """Call before importing model libraries; OS network isolation is separate."""
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
