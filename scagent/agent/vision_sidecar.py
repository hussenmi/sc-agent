"""
Vision sidecar for text-only main models.

When the main LLM cannot accept images (e.g. DeepSeek V4), figures produced by
tools are sent here. The sidecar — a separate, OpenAI-compatible vision model
configured via ``SCAGENT_VISION_MODEL`` — returns a structured textual
description that is injected into the main model's message stream in place of
the image.

Activation rule: only used when the main model lacks vision support AND
``SCAGENT_VISION_MODEL`` is set. Multimodal main models are unaffected.
"""

from __future__ import annotations

import base64 as _base64
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a scientific figure-description sidecar for a single-cell RNA-seq "
    "analysis agent. The main reasoning model is TEXT-ONLY and cannot see this "
    "figure. Your output will be inserted verbatim into its message stream in "
    "place of the image. Be faithful, concrete, and quantitative. Do not invent "
    "numbers; if uncertain, say so.\n\n"
    "Output the following sections in order, even if a section is empty:\n\n"
    "WHAT_THIS_IS: one sentence — plot type, axes, what each point/bar/cell represents.\n"
    "KEY_OBSERVATIONS: 3–6 bullets — dominant structure, separable groups, gradients, batch effects.\n"
    "NUMBERS_VISIBLE: any axis ranges, counts, thresholds, legend categories actually shown.\n"
    "ANOMALIES: anything off (empty clusters, label collisions, suspicious tails, missing legend).\n"
    "ACTIONABLE_FLAGS: bullets the main model should consider acting on, or \"none\".\n"
    "OPEN_QUESTIONS: things only follow-up inspection could resolve.\n"
)


_MULTI_IMAGE_NOTE = (
    "You are receiving {n} figures. For each figure, emit the standard sections "
    "under a \"=== FIGURE {{i}}: {{path}} ===\" header. After the last figure, "
    "append a final block:\n"
    "COMPARATIVE_NOTE: 2–4 bullets comparing the figures (only if they are "
    "clearly comparable, e.g. same plot type with different colorings). Otherwise "
    "write \"none\".\n"
)


@dataclass
class VisionSidecarConfig:
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 900
    timeout_s: float = 45.0


class VisionSidecar:
    """OpenAI-compatible vision describer used as a fallback for text-only main models."""

    _CACHE_CAP = 128

    def __init__(self, cfg: VisionSidecarConfig, log: Optional[logging.Logger] = None):
        self.cfg = cfg
        self._log = log or logger
        self._client = None  # lazy
        self._cache: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        # Which max-output kwarg this endpoint accepts. Newer OpenAI models
        # (gpt-5.x, o-series) require ``max_completion_tokens``; older models
        # (gpt-4o, gpt-3.5) require ``max_tokens``. Discovered on first call,
        # cached for the session.
        self._max_tokens_kwarg: Optional[str] = None

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_env(cls) -> Optional["VisionSidecar"]:
        """Construct from environment. Returns None when not configured.

        Required:  SCAGENT_VISION_MODEL
        Optional:  SCAGENT_VISION_BASE_URL, SCAGENT_VISION_API_KEY,
                   SCAGENT_VISION_MAX_TOKENS
        Falls back to OPENAI_API_KEY if SCAGENT_VISION_API_KEY is unset.
        """
        model = (os.environ.get("SCAGENT_VISION_MODEL") or "").strip()
        if not model:
            return None
        api_key = (
            os.environ.get("SCAGENT_VISION_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            logger.warning(
                "SCAGENT_VISION_MODEL=%s is set but no SCAGENT_VISION_API_KEY or "
                "OPENAI_API_KEY is available; vision sidecar disabled.",
                model,
            )
            return None
        base_url = (os.environ.get("SCAGENT_VISION_BASE_URL") or "").strip() or None
        try:
            max_tokens = int(os.environ.get("SCAGENT_VISION_MAX_TOKENS", "900"))
        except ValueError:
            max_tokens = 900
        cfg = VisionSidecarConfig(
            model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens
        )
        logger.info(
            "Vision sidecar enabled: model=%s base_url=%s",
            cfg.model, cfg.base_url or "<openai-default>",
        )
        return cls(cfg)

    # ---- public API ------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self.cfg.model

    def describe(
        self,
        images: List[Dict[str, Any]],
        world_state: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        comparative: bool = False,
    ) -> Dict[str, Any]:
        """Describe one or more figures with structured output.

        Parameters
        ----------
        images : list of dicts with at least {"base64", "mime", "path"} and
                 optionally {"role", "image_context"}.
        world_state : compact snapshot from AgentWorldState.snapshot() (or None).
        question : optional focused follow-up question for the sidecar.
        comparative : whether to ask for a COMPARATIVE_NOTE section across images.

        Returns
        -------
        dict with {status, text, model, latency_ms, cache_hits, n_images,
                   error (optional)}.
        """
        if not images:
            return {
                "status": "error",
                "text": "",
                "model": self.cfg.model,
                "latency_ms": 0,
                "cache_hits": 0,
                "n_images": 0,
                "error": "no images provided",
            }

        # Chunk >3 images into separate calls and concatenate.
        chunks = [images[i:i + 3] for i in range(0, len(images), 3)]
        all_texts: List[str] = []
        total_cache_hits = 0
        total_latency_ms = 0
        for chunk in chunks:
            r = self._describe_chunk(chunk, world_state, question, comparative)
            if r["status"] != "ok":
                # Surface first failure; partial results are unsafe to inject.
                return r
            all_texts.append(r["text"])
            total_cache_hits += r["cache_hits"]
            total_latency_ms += r["latency_ms"]
        return {
            "status": "ok",
            "text": "\n\n".join(all_texts),
            "model": self.cfg.model,
            "latency_ms": total_latency_ms,
            "cache_hits": total_cache_hits,
            "n_images": len(images),
        }

    # ---- internals -------------------------------------------------------

    def _call_with_token_kwarg_fallback(self, client, messages):
        """Issue the chat.completions call, picking the right max-tokens kwarg.

        Newer OpenAI models (gpt-5.x, o-series) require ``max_completion_tokens``
        and reject ``max_tokens`` with a 400 ``unsupported_parameter`` error.
        Older models only know ``max_tokens``. We try the modern kwarg first,
        fall back on the legacy one if the modern one is rejected, and cache
        the winning kwarg on the instance for the rest of the session.
        """
        kwargs_common = {
            "model": self.cfg.model,
            "messages": messages,
            "timeout": self.cfg.timeout_s,
        }
        order = (
            (self._max_tokens_kwarg,)
            if self._max_tokens_kwarg is not None
            else ("max_completion_tokens", "max_tokens")
        )
        last_exc: Optional[Exception] = None
        for kwarg in order:
            try:
                resp = client.chat.completions.create(
                    **kwargs_common,
                    **{kwarg: self.cfg.max_tokens},
                )
                self._max_tokens_kwarg = kwarg
                return resp
            except Exception as exc:
                msg = str(exc)
                # Only retry on the specific "wrong token-limit kwarg" 400.
                # Any other error (auth, rate limit, network) should bubble up.
                if (
                    "unsupported_parameter" in msg
                    or "max_tokens" in msg
                    or "max_completion_tokens" in msg
                ) and kwarg != order[-1]:
                    last_exc = exc
                    continue
                raise
        # Should be unreachable, but raise the captured error to be safe.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("vision sidecar: no token kwarg succeeded")

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "The 'openai' package is required for the vision sidecar."
                ) from exc
            kwargs: Dict[str, Any] = {"api_key": self.cfg.api_key}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _describe_chunk(
        self,
        images: List[Dict[str, Any]],
        world_state: Optional[Dict[str, Any]],
        question: Optional[str],
        comparative: bool,
    ) -> Dict[str, Any]:
        cache_key = self._cache_key(images, question)
        if cache_key in self._cache:
            # LRU touch
            text = self._cache.pop(cache_key)
            self._cache[cache_key] = text
            return {
                "status": "ok",
                "text": text,
                "model": self.cfg.model,
                "latency_ms": 0,
                "cache_hits": 1,
                "n_images": len(images),
            }

        system_prompt = _SYSTEM_PROMPT
        if len(images) > 1 or comparative:
            system_prompt = system_prompt + "\n" + _MULTI_IMAGE_NOTE.format(n=len(images))

        user_text = self._build_user_prompt(images, world_state, question)

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            mime = img.get("mime", "image/png")
            b64 = img.get("base64") or ""
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        t0 = time.time()
        try:
            client = self._get_client()
            resp = self._call_with_token_kwarg_fallback(client, messages)
        except Exception as exc:
            self._log.warning("Vision sidecar call failed: %s", exc)
            return {
                "status": "error",
                "text": "",
                "model": self.cfg.model,
                "latency_ms": int((time.time() - t0) * 1000),
                "cache_hits": 0,
                "n_images": len(images),
                "error": f"{type(exc).__name__}: {exc}",
            }

        latency_ms = int((time.time() - t0) * 1000)
        try:
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            self._log.warning("Vision sidecar returned unparseable response: %s", exc)
            return {
                "status": "error",
                "text": "",
                "model": self.cfg.model,
                "latency_ms": latency_ms,
                "cache_hits": 0,
                "n_images": len(images),
                "error": f"unparseable response: {exc}",
            }

        if not text:
            return {
                "status": "error",
                "text": "",
                "model": self.cfg.model,
                "latency_ms": latency_ms,
                "cache_hits": 0,
                "n_images": len(images),
                "error": "empty response",
            }

        # Cache and bound size.
        self._cache[cache_key] = text
        while len(self._cache) > self._CACHE_CAP:
            self._cache.popitem(last=False)

        return {
            "status": "ok",
            "text": text,
            "model": self.cfg.model,
            "latency_ms": latency_ms,
            "cache_hits": 0,
            "n_images": len(images),
        }

    # ---- prompt helpers --------------------------------------------------

    @staticmethod
    def _norm_question(question: Optional[str]) -> str:
        return " ".join((question or "").split()).lower()

    def _cache_key(
        self, images: List[Dict[str, Any]], question: Optional[str]
    ) -> Tuple[str, str]:
        h = hashlib.sha256()
        for img in images:
            b64 = (img.get("base64") or "").encode("utf-8", errors="ignore")
            h.update(hashlib.sha256(b64).digest())
            h.update(b"|")
        return (h.hexdigest(), self._norm_question(question))

    @staticmethod
    def _format_world_state(world_state: Optional[Dict[str, Any]]) -> str:
        if not world_state:
            return "(unavailable)"
        ds = (world_state.get("data_summary") or {})
        caps = (ds.get("capabilities") or {})
        proc = (ds.get("processing") or {})
        shape = ds.get("shape") or {}
        if isinstance(shape, dict):
            shape_str = f"{shape.get('n_obs','?')} cells x {shape.get('n_vars','?')} genes"
        else:
            shape_str = str(shape)

        batch_keys = ds.get("batch_keys") or ds.get("batch_columns") or []
        annotation_cols = caps.get("annotation_keys") or []
        cluster_keys = caps.get("cluster_keys") or []
        primary_cluster = (
            ds.get("primary_cluster_key") or (cluster_keys[0] if cluster_keys else None)
        )
        n_clusters = ds.get("n_clusters")

        step_log = world_state.get("step_log") or []
        recent = []
        for entry in step_log[-3:]:
            if isinstance(entry, dict):
                tool = entry.get("tool") or entry.get("action") or "?"
                status = entry.get("status") or ""
                recent.append(f"{tool} ({status})")
            else:
                recent.append(str(entry))

        proc_flags = ", ".join(
            f"{k}={v}" for k, v in proc.items() if isinstance(v, bool) and v
        ) or "(none)"

        lines = [
            f"  analysis_stage: {world_state.get('analysis_stage')}",
            f"  shape: {shape_str}",
            f"  processed flags: {proc_flags}",
            f"  primary cluster key: {primary_cluster}",
            f"  n_clusters: {n_clusters if n_clusters is not None else '?'}",
            f"  cluster keys: {', '.join(cluster_keys[:5]) or '(none)'}",
            f"  annotation columns: {', '.join(annotation_cols[:5]) or '(none)'}",
            f"  batch keys: {', '.join(map(str, batch_keys)) or '(none)'}",
            f"  recent steps: {' | '.join(recent) or '(none)'}",
        ]
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        images: List[Dict[str, Any]],
        world_state: Optional[Dict[str, Any]],
        question: Optional[str],
    ) -> str:
        ws_block = self._format_world_state(world_state)

        figure_blocks: List[str] = []
        for i, img in enumerate(images, start=1):
            ctx = img.get("image_context") or {}
            path = img.get("path") or ctx.get("output_path") or "(unknown path)"
            role = img.get("role") or "figure"
            plot_type = ctx.get("plot_type") or "—"
            color_by = ctx.get("color_by") or "—"
            cluster_key = ctx.get("cluster_key") or "—"
            tool = ctx.get("producing_tool") or ctx.get("tool") or "—"
            header = f"=== FIGURE {i}: {path} ===" if len(images) > 1 else f"FIGURE: {path}"
            figure_blocks.append(
                "\n".join([
                    header,
                    f"  role: {role}",
                    f"  plot_type: {plot_type}",
                    f"  color_by: {color_by}",
                    f"  cluster_key: {cluster_key}",
                    f"  producing_tool: {tool}",
                ])
            )

        q_line = (question or "").strip() or "—"
        return (
            "ANALYSIS CONTEXT (compact world_state snapshot):\n"
            f"{ws_block}\n\n"
            "FIGURE CONTEXT (from the producing tool):\n"
            + "\n\n".join(figure_blocks) + "\n\n"
            f"USER FOLLOW-UP QUESTION (optional): {q_line}\n"
        )


def encode_image_path_to_b64(path: str) -> Tuple[str, str]:
    """Read an image file and return (base64_str, mime). Mime inferred from suffix."""
    with open(path, "rb") as f:
        b64 = _base64.b64encode(f.read()).decode("utf-8")
    ext = (os.path.splitext(path)[1] or "").lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext, "image/png")
    return b64, mime
