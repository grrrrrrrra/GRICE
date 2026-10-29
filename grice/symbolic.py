"""
Symbolic maxim scores.

All scores are violation scores in [0, 1]: higher means stronger evidence that
the maxim is violated. Inverted features are marked in the code.

    QUANTITY : wateriness, lexical diversity (1 - TTR)
               variant v2: referent ambiguity, modifier sparsity
    QUALITY  : vagueness, contradiction detection
    RELATION : semantic relevance (inverted), grounding confidence (inverted)
    MANNER   : polysemy, syntactic depth, semantic density (inverted)
               variant v2: depth relative to sentence length

Requires spaCy (en_core_web_sm) and NLTK (wordnet, stopwords). RELATION uses
sentence-transformers when available and TF-IDF cosine otherwise; the backend
is recorded in the run manifest.
"""

from __future__ import annotations

import ast
import functools
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAXIMS = ("quantity", "quality", "relation", "manner")

# --- lexicons -------------------------------------------------------------
HEDGE_WORDS = {
    "something", "somewhere", "somebody", "someone", "somehow", "anything",
    "anywhere", "stuff", "thing", "things", "thingy", "object", "item",
    "maybe", "perhaps", "possibly", "probably", "might", "may", "could",
    "roughly", "approximately", "about", "kind", "sort", "some", "few",
    "several", "whatever", "whichever", "wherever", "appropriate", "suitable",
    "proper", "usual", "normal", "regular", "nice", "good", "bad",
}
EXISTENTIAL_RE = re.compile(r"\bthere\s+(?:is|are|was|were)\s+(?:a|an|the|some)?\s*([\w\s\-]{2,40})",
                            flags=re.IGNORECASE)

# normalisation caps: values above the cap saturate at 1.0
POLYSEMY_CAP = 8.0        # mean WordNet synsets per content word
DEPTH_CAP = 8.0           # max dependency-tree depth


# --- lazy singletons ------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


@functools.lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    import nltk
    from nltk.corpus import stopwords
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


@functools.lru_cache(maxsize=1)
def _wn():
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return wn


@functools.lru_cache(maxsize=20000)
def _synset_count(lemma: str, pos: Optional[str] = None) -> int:
    wn = _wn()
    return len(wn.synsets(lemma, pos=pos) if pos else wn.synsets(lemma))


@functools.lru_cache(maxsize=20000)
def _antonyms(lemma: str) -> frozenset:
    wn = _wn()
    out = set()
    for syn in wn.synsets(lemma):
        for lem in syn.lemmas():
            for ant in lem.antonyms():
                out.add(ant.name().lower().replace("_", " "))
    return frozenset(out)


# --- context handling -----------------------------------------------------
def parse_context(raw) -> Tuple[str, List[str]]:
    """Return (flat text, list of entity strings) for an environment field.

    Handles: python/json list literals, comma-separated strings, free text.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "", []
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
        return ", ".join(items), items
    s = str(raw).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(s)
            if isinstance(parsed, (list, tuple)):
                items = [str(x) for x in parsed]
                return ", ".join(items), items
            if isinstance(parsed, dict):
                items = [f"{k}: {v}" for k, v in parsed.items()]
                return ", ".join(items), items
        except Exception:
            pass
    items = [p.strip() for p in re.split(r"[,;\n]", s) if p.strip()]
    return s, items or ([s] if s else [])


def _entity_lemmas(entities: Sequence[str]) -> List[set]:
    """Lemma set for each environment entity."""
    nlp = _nlp()
    out = []
    for ent in entities:
        doc = nlp(str(ent).lower())
        out.append({t.lemma_ for t in doc if not t.is_punct and not t.is_space})
    return out


# =========================================================================
# QUANTITY
# =========================================================================
def quantity_features(doc) -> Dict[str, float]:
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return {"qt_wateriness": 0.0, "qt_low_diversity": 0.0}
    stops = _stopwords()
    words = [t.text.lower() for t in toks]

    # Feature 1: wateriness — proportion of stopwords (high = low info density)
    wateriness = sum(w in stops for w in words) / len(words)

    # Feature 2: inverse type-token ratio (low diversity = underspecification)
    ttr = len(set(words)) / len(words)
    low_diversity = 1.0 - ttr

    return {"qt_wateriness": float(wateriness), "qt_low_diversity": float(low_diversity)}


def quantity_features_v2(doc, entities, ent_lemmas) -> Dict[str, float]:
    """Quantity features computed against the environment (variant v2).

    Feature 1 (referent ambiguity): share of grounded noun phrases that match
      more than one environment entity.
    Feature 2 (modifier sparsity): 1 - mean number of modifiers per noun chunk,
      capped at 3.
    """
    chunks = list(doc.noun_chunks)
    if not chunks:
        return {"qt_referent_ambiguity": 0.0, "qt_modifier_sparsity": 0.0}

    stops = _stopwords()
    multi, grounded = 0, 0
    for chunk in chunks:
        terms = {t.lemma_.lower() for t in chunk
                 if not t.is_punct and not t.is_space
                 and t.pos_ not in {"DET", "PRON"}
                 and t.lemma_.lower() not in stops}
        if not terms:
            continue
        # an entity matches if it contains every content lemma of the phrase
        matches = sum(1 for lem in ent_lemmas if terms <= lem)
        if matches == 0:
            continue                    # ungrounded phrases are scored under RELATION
        grounded += 1
        if matches > 1:
            multi += 1
    referent_ambiguity = multi / grounded if grounded else 0.0

    MOD_DEPS = {"amod", "compound", "nummod", "poss", "prep", "relcl", "acl"}
    mods = []
    for chunk in chunks:
        n = sum(1 for t in chunk.root.children if t.dep_ in MOD_DEPS)
        n += sum(1 for t in chunk if t.dep_ in {"amod", "compound", "nummod"}
                 and t != chunk.root)
        mods.append(min(n, 3) / 3.0)
    modifier_sparsity = 1.0 - float(np.mean(mods))

    return {"qt_referent_ambiguity": float(referent_ambiguity),
            "qt_modifier_sparsity": float(modifier_sparsity)}


# =========================================================================
# QUALITY
# =========================================================================
def _property_in_context(adj: str, noun_lemma: str, entities: Sequence[str],
                         ent_lemmas: Sequence[set]) -> float:
    """Score a single adjective-noun pair against the environment.

    0.0  property explicitly present on a matching object
    0.5  noun absent from context, or present but without the property
    1.0  an antonym of the property is asserted for that object
    """
    hits = [i for i, lem in enumerate(ent_lemmas) if noun_lemma in lem]
    if not hits:
        return 0.5  # (a) noun existence in context
    ants = _antonyms(adj)
    partial = False
    for i in hits:
        ent_text = str(entities[i]).lower()
        if adj in ent_text:
            return 0.0  # (b) property present
        if any(a in ent_text for a in ants):
            return 1.0  # (c) antonym conflict
        partial = True
    return 0.5 if partial else 0.5


def quality_features(doc, entities: Sequence[str], ent_lemmas: Sequence[set]) -> Dict[str, float]:
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return {"ql_vagueness": 0.0, "ql_contradiction": 0.0}

    # Feature 1: vagueness — proportion of hedging words
    vagueness = sum(t.lemma_.lower() in HEDGE_WORDS or t.text.lower() in HEDGE_WORDS
                    for t in toks) / len(toks)

    # Feature 2: contradiction detection
    scores: List[float] = []
    for tok in doc:
        if tok.pos_ == "ADJ" and tok.dep_ in {"amod", "acomp"} and tok.head.pos_ in {"NOUN", "PROPN"}:
            scores.append(_property_in_context(tok.lemma_.lower(), tok.head.lemma_.lower(),
                                               entities, ent_lemmas))
    # existential claims: "there is X" checked against context
    for m in EXISTENTIAL_RE.finditer(doc.text):
        phrase = m.group(1).lower().strip()
        head = phrase.split()[-1] if phrase.split() else ""
        if head and not any(head in lem for lem in ent_lemmas):
            scores.append(1.0)

    contradiction = float(max(scores)) if scores else 0.0
    return {"ql_vagueness": float(vagueness), "ql_contradiction": contradiction}


# =========================================================================
# RELATION
# =========================================================================
class RelevanceScorer:
    """Cosine similarity between instruction and context embeddings.

    Uses sentence-transformers when available, otherwise TF-IDF cosine; the
    backend in use is stored in `self.backend`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", allow_fallback: bool = True,
                 device: str = "cpu"):
        self.backend = "none"
        self._model = None
        self._tfidf = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self.backend = f"sentence-transformers/{model_name}"
        except Exception:
            if not allow_fallback:
                raise
            self.backend = "tfidf-fallback"

    def fit_fallback(self, corpus: Sequence[str]) -> None:
        if self.backend != "tfidf-fallback":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(list(corpus))

    def similarity(self, instruction: str, context: str) -> float:
        if not context.strip():
            return 0.0
        if self._model is not None:
            a, b = self._model.encode([instruction, context], convert_to_numpy=True)
        else:
            if self._tfidf is None:
                self.fit_fallback([instruction, context])
            m = self._tfidf.transform([instruction, context]).toarray()
            a, b = m[0], m[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)


def relation_features(doc, context_text: str, entities: Sequence[str],
                      ent_lemmas: Sequence[set], scorer: RelevanceScorer) -> Dict[str, float]:
    # Feature 1: semantic relevance, inverted -> violation
    sim = scorer.similarity(doc.text, context_text)
    irrelevance = float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))

    # Feature 2: grounding confidence — share of NPs matched to context entities
    chunks = [c for c in doc.noun_chunks]
    if not chunks or not ent_lemmas:
        ungrounded = 1.0 if (chunks and not ent_lemmas) else 0.0
    else:
        matched = 0
        for chunk in chunks:
            head = chunk.root.lemma_.lower()
            if any(head in lem for lem in ent_lemmas):
                matched += 1
        ungrounded = 1.0 - matched / len(chunks)  # inverted -> violation

    return {"rel_irrelevance": irrelevance, "rel_ungrounded": float(ungrounded)}


# =========================================================================
# MANNER
# =========================================================================
_WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def _tree_depth(token, _seen=None) -> int:
    if _seen is None:
        _seen = set()
    if token.i in _seen:
        return 0
    _seen.add(token.i)
    children = [c for c in token.children if c.i not in _seen]
    if not children:
        return 1
    return 1 + max(_tree_depth(c, _seen) for c in children)


def manner_features(doc, relative_depth: bool = False) -> Dict[str, float]:
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return {"mn_polysemy": 0.0, "mn_depth": 0.0, "mn_low_density": 0.0}

    # Feature 1: polysemy — mean WordNet synsets per content word, normalised
    counts = [_synset_count(t.lemma_.lower(), _WN_POS[t.pos_])
              for t in toks if t.pos_ in _WN_POS]
    polysemy = float(np.clip(np.mean(counts) / POLYSEMY_CAP, 0.0, 1.0)) if counts else 0.0

    # Feature 2: maximum dependency-tree depth, normalised
    depths = [_tree_depth(sent.root) for sent in doc.sents]
    raw_depth = max(depths) if depths else 0
    if relative_depth:
        # depth normalised by the expected depth for a sentence of this length
        expected = max(1.0, np.log2(len(toks) + 1))
        depth = float(np.clip((raw_depth / expected) / 2.0, 0.0, 1.0))
    else:
        depth = float(np.clip(raw_depth / DEPTH_CAP, 0.0, 1.0))

    # Feature 3: semantic density, inverted -> violation
    content = sum(t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} for t in toks)
    low_density = 1.0 - content / len(toks)

    return {"mn_polysemy": polysemy, "mn_depth": depth, "mn_low_density": float(low_density)}


# =========================================================================
# aggregation
# =========================================================================
@dataclass
class SymbolicScores:
    v_sym_quantity: float
    v_sym_quality: float
    v_sym_relation: float
    v_sym_manner: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def symbolic_scores(instruction: str, context_raw, scorer: RelevanceScorer,
                    return_subfeatures: bool = True,
                    variant: str = "paper") -> Dict[str, float]:
    """Symbolic scores for one (instruction, context) pair.

    Each maxim score is the unweighted mean of its sub-features.
    """
    doc = _nlp()(str(instruction))
    context_text, entities = parse_context(context_raw)
    ent_lemmas = _entity_lemmas(entities)

    if variant == "paper":
        qt = quantity_features(doc)
        mn = manner_features(doc, relative_depth=False)
    elif variant == "v2":
        qt = quantity_features_v2(doc, entities, ent_lemmas)
        mn = manner_features(doc, relative_depth=True)
    else:
        raise ValueError(f"unknown feature variant: {variant}")
    ql = quality_features(doc, entities, ent_lemmas)
    rel = relation_features(doc, context_text, entities, ent_lemmas, scorer)

    out: Dict[str, float] = {
        "v_sym_quantity": float(np.mean(list(qt.values()))),
        "v_sym_quality": float(np.mean(list(ql.values()))),
        "v_sym_relation": float(np.mean(list(rel.values()))),
        "v_sym_manner": float(np.mean(list(mn.values()))),
    }
    if return_subfeatures:
        out.update(qt)
        out.update(ql)
        out.update(rel)
        out.update(mn)
    return out
