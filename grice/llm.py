"""
LLM backends for the per-maxim assessment.

Only newly generated tokens are decoded. An answer that cannot be parsed as
YES/NO is returned as NaN, and parse failures are counted in GenerationStats.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

YES_RE = re.compile(r"^\W*(yes|no)\b", flags=re.IGNORECASE)
ANY_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def parse_yes_no(text: str) -> Optional[int]:
    """1 for YES, 0 for NO, None if the answer is not a clean yes/no.

    Uses the leading token if possible, otherwise the first standalone yes/no
    in the generated text.
    """
    if text is None:
        return None
    s = text.strip()
    m = YES_RE.match(s) or ANY_RE.search(s)
    if not m:
        return None
    return 1 if m.group(1).lower() == "yes" else 0


# =========================================================================
# cache
# =========================================================================
class Cache:
    """SQLite cache keyed by (model, prompt)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha1(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM cache WHERE k=?",
                                (self.key(model, prompt),)).fetchone()
        return row[0] if row else None

    def put(self, model: str, prompt: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                          (self.key(model, prompt), value))
        self.conn.commit()


# =========================================================================
# backends
# =========================================================================
@dataclass
class GenerationStats:
    n_calls: int = 0
    n_cached: int = 0
    n_parse_fail: int = 0
    n_api_error: int = 0
    failures: List[str] = field(default_factory=list)

    def report(self) -> Dict[str, float]:
        total = max(self.n_calls, 1)
        return {
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "parse_failure_rate": round(self.n_parse_fail / total, 4),
            "api_error_rate": round(self.n_api_error / total, 4),
        }


class BaseBackend:
    name = "base"

    def __init__(self, model_name: str, cache_path: Optional[str] = None,
                 max_new_tokens: int = 8):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.cache = Cache(cache_path) if cache_path else None
        self.stats = GenerationStats()

    def _generate(self, prompt: str, system: Optional[str]) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self.cache is not None:
            hit = self.cache.get(self.model_name, prompt)
            if hit is not None:
                self.stats.n_cached += 1
                self.stats.n_calls += 1
                return hit
        try:
            out = self._generate(prompt, system)
        except Exception as exc:  # noqa: BLE001
            self.stats.n_api_error += 1
            self.stats.n_calls += 1
            self.stats.failures.append(f"API: {exc}")
            return ""
        self.stats.n_calls += 1
        if self.cache is not None:
            self.cache.put(self.model_name, prompt, out)
        return out

    def yes_no(self, prompt: str, system: Optional[str] = None) -> float:
        raw = self.generate(prompt, system)
        val = parse_yes_no(raw)
        if val is None:
            self.stats.n_parse_fail += 1
            if len(self.stats.failures) < 25:
                self.stats.failures.append(repr(raw[:120]))
            return float("nan")
        return float(val)


YES_VARIANTS = ["YES", "Yes", "yes", " YES", " Yes", " yes"]
NO_VARIANTS = ["NO", "No", "no", " NO", " No", " no"]


class HFBackend(BaseBackend):
    """Local Hugging Face transformers backend for causal LMs.

    `enable_thinking` is passed to the chat template when it is supported
    (Qwen3). For chat templates without a system role (e.g. Gemma), the system
    prompt is merged into the user turn.
    """

    name = "hf"

    def __init__(self, model_name: str, cache_path: Optional[str] = None,
                 max_new_tokens: int = 8, load_in_4bit: bool = False,
                 dtype: str = "bfloat16", enable_thinking: bool = False):
        super().__init__(model_name, cache_path, max_new_tokens)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # bfloat16 requires compute capability >= 8.0; use float16 otherwise
        if dtype == "bfloat16" and torch.cuda.is_available() \
                and not torch.cuda.is_bf16_supported():
            cc = torch.cuda.get_device_capability(0)
            print(f"note: GPU compute capability {cc[0]}.{cc[1]} has no bfloat16; "
                  f"using float16 instead")
            dtype = "float16"
        self.dtype = dtype

        kwargs: Dict = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self._yes_ids = self._first_token_ids(YES_VARIANTS)
        self._no_ids = self._first_token_ids(NO_VARIANTS)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(f"cannot locate YES/NO tokens for {model_name}")

    def _first_token_ids(self, variants) -> List[int]:
        ids = set()
        for v in variants:
            enc = self.tokenizer.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _chat_text(self, prompt: str, system: Optional[str]) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        for attempt in (0, 1, 2):
            try:
                kw = {"tokenize": False, "add_generation_prompt": True}
                if attempt == 0:
                    kw["enable_thinking"] = self.enable_thinking   # Qwen3
                if attempt == 2:                                   # Gemma: no system role
                    merged = f"{system}\n\n{prompt}" if system else prompt
                    messages = [{"role": "user", "content": merged}]
                return self.tokenizer.apply_chat_template(messages, **kw)
            except Exception:
                continue
        return f"{system}\n\n{prompt}" if system else prompt

    def _generate(self, prompt: str, system: Optional[str]) -> str:
        text = self._chat_text(prompt, system)
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                     # greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # decode only the generated continuation
        gen = out[0][enc["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def yes_no_prob(self, prompt: str, system: Optional[str] = None) -> float:
        """P(YES) / (P(YES) + P(NO)) from the next-token distribution."""
        if self.cache is not None:
            hit = self.cache.get(self.model_name, "[PROB]" + prompt)
            if hit is not None:
                self.stats.n_cached += 1
                self.stats.n_calls += 1
                return float(hit)
        try:
            text = self._chat_text(prompt, system)
            enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits[0, -1]
            # float32 before softmax to avoid overflow in fp16
            logits = logits.float()
            if not self.torch.isfinite(logits).all():
                self.stats.n_parse_fail += 1
                self.stats.n_calls += 1
                self.stats.failures.append("PROB: non-finite logits (fp16 overflow)")
                return float("nan")
            probs = self.torch.softmax(logits, dim=-1)
            p_yes = float(probs[self._yes_ids].sum())
            p_no = float(probs[self._no_ids].sum())
        except Exception as exc:  # noqa: BLE001
            self.stats.n_api_error += 1
            self.stats.n_calls += 1
            self.stats.failures.append(f"PROB: {exc}")
            return float("nan")
        self.stats.n_calls += 1
        total = p_yes + p_no
        if total <= 0:
            self.stats.n_parse_fail += 1
            return float("nan")
        val = p_yes / total
        # total probability mass on the YES/NO tokens
        self.last_answer_mass = total
        if self.cache is not None:
            self.cache.put(self.model_name, "[PROB]" + prompt, repr(val))
        return val


class OpenAIBackend(BaseBackend):
    """OpenAI Chat Completions backend (e.g. gpt-5-mini).

    Reasoning models (gpt-5, o1, o3, o4) get a completion budget of 512 tokens
    and `reasoning_effort` (default 'minimal'); on an empty answer the budget
    is doubled, up to 4096.
    """

    name = "openai"

    def __init__(self, model_name: str, cache_path: Optional[str] = None,
                 max_new_tokens: int = 8, max_retries: int = 4,
                 reasoning_effort: str = "minimal"):
        super().__init__(model_name, cache_path, max_new_tokens)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.max_retries = max_retries
        self.is_reasoning = model_name.startswith(("gpt-5", "o1", "o3", "o4"))
        self.reasoning_effort = reasoning_effort
        self.budget = 512 if self.is_reasoning else max(max_new_tokens, 4)

    def _kwargs(self, messages) -> Dict:
        kw: Dict = {"model": self.model_name, "messages": messages}
        if self.is_reasoning:
            kw["max_completion_tokens"] = self.budget
            kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["max_tokens"] = self.budget
            kw["temperature"] = 0.0
        return kw

    def _messages(self, prompt: str, system: Optional[str]):
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _generate(self, prompt: str, system: Optional[str]) -> str:
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    **self._kwargs(self._messages(prompt, system)))
                content = (resp.choices[0].message.content or "").strip()
                if not content and self.is_reasoning:
                    # empty answer: retry with a larger budget
                    self.budget = min(self.budget * 2, 4096)
                    last = RuntimeError("empty content (reasoning budget)")
                    continue
                return content
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"openai failed after {self.max_retries} retries: {last}")

    def yes_no_prob(self, prompt: str, system: Optional[str] = None) -> float:
        """P(YES) from token logprobs; falls back to the YES/NO answer when
        logprobs are unavailable."""
        if self.cache is not None:
            hit = self.cache.get(self.model_name, "[PROB]" + prompt)
            if hit is not None:
                self.stats.n_cached += 1
                self.stats.n_calls += 1
                return float(hit)
        try:
            kw = self._kwargs(self._messages(prompt, system))
            kw.update({"logprobs": True, "top_logprobs": 20})
            resp = self.client.chat.completions.create(**kw)
            top = resp.choices[0].logprobs.content[0].top_logprobs
            p_yes = sum(np.exp(t.logprob) for t in top
                        if t.token.strip().lower().startswith("yes"))
            p_no = sum(np.exp(t.logprob) for t in top
                       if t.token.strip().lower().startswith("no"))
        except Exception:
            self.n_logprob_fallback = getattr(self, "n_logprob_fallback", 0) + 1
            return self.yes_no(prompt, system)
        self.stats.n_calls += 1
        total = p_yes + p_no
        if total <= 0:
            self.stats.n_parse_fail += 1
            return float("nan")
        val = float(p_yes / total)
        if self.cache is not None:
            self.cache.put(self.model_name, "[PROB]" + prompt, repr(val))
        return val


class MockBackend(BaseBackend):
    """Deterministic hash-based backend for testing the pipeline."""

    name = "mock"

    def _generate(self, prompt: str, system: Optional[str]) -> str:
        h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
        return "YES" if h % 2 else "NO"

    def yes_no_prob(self, prompt: str, system: Optional[str] = None) -> float:
        h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
        self.stats.n_calls += 1
        return (h % 1000) / 1000.0


def build_backend(spec: Dict) -> BaseBackend:
    kind = spec.get("backend", "hf")
    common = dict(model_name=spec["model_name"],
                  cache_path=spec.get("cache_path"),
                  max_new_tokens=spec.get("max_new_tokens", 8))
    if kind == "hf":
        return HFBackend(load_in_4bit=spec.get("load_in_4bit", False),
                         dtype=spec.get("dtype", "bfloat16"),
                         enable_thinking=spec.get("enable_thinking", False),
                         **common)
    if kind == "openai":
        return OpenAIBackend(reasoning_effort=spec.get("reasoning_effort", "minimal"),
                             **common)
    if kind == "mock":
        return MockBackend(**common)
    raise ValueError(f"unknown backend: {kind}")


# =========================================================================
# assessment driver
# =========================================================================
def assess_maxims(backend: BaseBackend, instruction: str, context: str,
                  score_mode: str = "both",
                  prompt_version="v1") -> Dict[str, float]:
    """Assess each maxim with a separate YES/NO prompt.

    score_mode:
        'binary'  v_llm_<maxim> is the parsed answer (1 = YES, 0 = NO)
        'prob'    v_llm_<maxim> is P(YES)
        'both'    v_llm_<maxim> is P(YES) and v_llm_bin_<maxim> is the parsed answer
    """
    from .prompts import MAXIMS_ORDER, maxim_prompt, resolve_mix, SYSTEM_PROMPT
    versions = (prompt_version if isinstance(prompt_version, dict)
                else resolve_mix(prompt_version))
    out: Dict[str, float] = {}
    for maxim in MAXIMS_ORDER:
        prompt = maxim_prompt(maxim, instruction, context, version=versions[maxim])
        if score_mode in ("binary", "both"):
            key = "v_llm_bin_" if score_mode == "both" else "v_llm_"
            out[f"{key}{maxim}"] = backend.yes_no(prompt, SYSTEM_PROMPT)
        if score_mode in ("prob", "both"):
            out[f"v_llm_{maxim}"] = backend.yes_no_prob(prompt, SYSTEM_PROMPT)
    return out
