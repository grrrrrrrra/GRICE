"""
Per-maxim prompt templates.

Each maxim is assessed with a separate prompt that asks for a YES/NO answer.
Three versions are available:
    v1  maxim definition and one violation example
    v2  maxim definition with one violation and one non-violation example
    v3  an operational question that does not name the maxim, with one example
The same templates are used for every backbone.
"""

from typing import Dict

MAXIMS_ORDER = ("quantity", "quality", "relation", "manner")

MAXIM_PROMPTS: Dict[str, Dict[str, str]] = {
    "quantity": {
        "definition": ("The instruction provides too little or too much information "
                       "for the listener to uniquely identify what to do."),
        "example_instruction": "Bring me the book",
        "example_context": "There are three books on the table",
        "example_answer": "YES",
    },
    "quality": {
        "definition": ("The instruction contains false, impossible, or incorrect "
                       "assumptions about the world or the context."),
        "example_instruction": "Take the clean cup",
        "example_context": "All cups are dirty",
        "example_answer": "YES",
    },
    "relation": {
        "definition": "The instruction is not relevant to the current task or context.",
        "example_instruction": "Turn on the light",
        "example_context": "The task is to set the table",
        "example_answer": "YES",
    },
    "manner": {
        "definition": ("The instruction is unclear, vague, or structurally ambiguous, "
                       "making it difficult to interpret."),
        "example_instruction": "Open the window",
        "example_context": "The room has both a physical window and a software window",
        "example_answer": "YES",
    },
}

# -------------------------------------------------------------------------
# v2: a non-violation example for every maxim; QUALITY is defined as
# presupposition failure relative to the environment
# -------------------------------------------------------------------------
MAXIM_PROMPTS_V2: Dict[str, Dict[str, str]] = {
    "quantity": {
        "definition": ("The instruction does not give enough information to identify a "
                       "single object or a single action in this environment. If several "
                       "objects in the environment fit the description equally well, the "
                       "maxim is violated."),
        "yes_instruction": "Bring me the book",
        "yes_context": "book A, book B, book C, table",
        "no_instruction": "Bring me the red book from the top shelf",
        "no_context": "red book on the top shelf, blue book, table",
    },
    "quality": {
        "definition": ("The instruction takes for granted something that is not true of "
                       "this environment: it refers to an object that is not present, or "
                       "requires a property that no available object has. If everything "
                       "the instruction presupposes is actually there, the maxim is not "
                       "violated, however vague the instruction may be."),
        "yes_instruction": "Take the clean cup",
        "yes_context": "dirty cup, dirty mug, sink",
        "no_instruction": "Take the cup",
        "no_context": "clean cup, glass, sink",
    },
    "relation": {
        "definition": ("The instruction asks for something unconnected to the current task "
                       "or to the objects available. An unusual but task-connected step is "
                       "not a violation."),
        "yes_instruction": "Turn on the light",
        "yes_context": "plate, fork, napkin; the task is to set the table",
        "no_instruction": "Bring the salt",
        "no_context": "salt, pepper, soup pot; the task is to season the soup",
    },
    "manner": {
        "definition": ("The wording itself is unclear: a word has two readings, a phrase "
                       "can attach in more than one place, or the sentence is garbled. If "
                       "the sentence is clearly phrased and merely leaves out information, "
                       "that is QUANTITY, not MANNER."),
        "yes_instruction": "Put the cup on the table in the kitchen",
        "yes_context": "cup, kitchen table, corner table",
        "no_instruction": "Put the cup on the kitchen table",
        "no_context": "cup, kitchen table, corner table",
    },
}

_TEMPLATE_V2 = """Determine whether the instruction violates the maxim of {maxim_upper}.

{definition}

Judge the instruction against the environment given below, and nothing else.
Do not assume objects that are not listed.

Example of a violation:
Instruction: "{yes_instruction}"
Environment: {yes_context}
Answer: YES

Example of NO violation:
Instruction: "{no_instruction}"
Environment: {no_context}
Answer: NO

Now evaluate:
Instruction: "{instruction}"
Environment: {context}

Answer with a single word, YES or NO. Do not explain.
Answer:"""


# -------------------------------------------------------------------------
# v3: each maxim is restated as an operational question about the situation;
# the prompt does not name the maxim
# -------------------------------------------------------------------------
MAXIM_PROMPTS_V3: Dict[str, Dict[str, str]] = {
    "quantity": {
        "question": ("Does the instruction leave out a detail the robot would need in "
                     "order to act — which object to use, how much, or which of several "
                     "similar objects is meant?"),
        "example_instruction": "Bring me the book",
        "example_context": "book A, book B, book C, table",
        "example_answer": "YES",
    },
    "quality": {
        "question": ("Does the instruction assume something that is not true of this "
                     "environment — an object that is not there, or a property that none "
                     "of the available objects has?"),
        "example_instruction": "Take the clean cup",
        "example_context": "dirty cup, dirty mug, sink",
        "example_answer": "YES",
    },
    "relation": {
        "question": ("Is the requested action unrelated to what the robot is evidently "
                     "supposed to be doing with these objects?"),
        "example_instruction": "Turn on the light",
        "example_context": "plate, fork, napkin; the task is to set the table",
        "example_answer": "YES",
    },
    "manner": {
        "question": ("Can this sentence be understood in more than one way — a word with "
                     "two meanings, or a phrase that could attach in two places?"),
        "example_instruction": "Put the cup on the table in the kitchen",
        "example_context": "cup, kitchen table, corner table",
        "example_answer": "YES",
    },
}

_TEMPLATE_V3 = """You are helping a household robot decide whether it can carry out an
instruction without asking the person anything.

Environment: {context}
Instruction: "{instruction}"

Question: {question}

Example:
Instruction: "{example_instruction}"
Environment: {example_context}
Answer: {example_answer}

Answer with a single word, YES or NO. Do not explain.
Answer:"""


_TEMPLATE = """Determine whether the instruction violates the maxim of {maxim_upper}.

{definition}

Example of a clear violation:
Instruction: "{example_instruction}"
Context: {example_context}
Answer: {example_answer}

Now evaluate:
Instruction: "{instruction}"
Context: {context}

Answer with a single word, YES or NO. Do not explain.
Answer:"""

SYSTEM_PROMPT = ("You are an expert in pragmatics and Grice's conversational maxims. "
                 "You answer with exactly one word: YES or NO.")


# Prompt version per maxim for --prompt-version mix (used in the paper),
# selected on the development groups listed in out/dev_groups.txt;
# scripts/03_experiments.py --exclude-groups removes them from evaluation.
DEFAULT_PROMPT_MIX = {
    "quantity": "v1",
    "quality": "v1",
    "relation": "v1",
    "manner": "v3",
}


def resolve_mix(spec: str) -> Dict[str, str]:
    """Parse a prompt-version spec.

    'v1'                     -> every maxim uses v1
    'mix'                    -> DEFAULT_PROMPT_MIX
    'quantity=v1,manner=v3'  -> explicit, unlisted maxims fall back to v1
    """
    spec = (spec or "v1").strip()
    if spec == "mix":
        return dict(DEFAULT_PROMPT_MIX)
    if "=" not in spec:
        return {m: spec for m in MAXIMS_ORDER}
    out = {m: "v1" for m in MAXIMS_ORDER}
    for part in spec.split(","):
        maxim, _, version = part.partition("=")
        maxim, version = maxim.strip(), version.strip()
        if maxim not in out:
            raise ValueError(f"unknown maxim in prompt spec: {maxim}")
        if version not in ("v1", "v2", "v3"):
            raise ValueError(f"unknown prompt version: {version}")
        out[maxim] = version
    return out


def maxim_prompt(maxim: str, instruction: str, context: str,
                 version: str = "v1") -> str:
    """Prompt for one maxim; `version` is 'v1', 'v2' or 'v3'."""
    if version == "v3":
        if maxim not in MAXIM_PROMPTS_V3:
            raise KeyError(f"unknown maxim: {maxim}")
        return _TEMPLATE_V3.format(instruction=instruction, context=context,
                                   **MAXIM_PROMPTS_V3[maxim])
    if version == "v2":
        if maxim not in MAXIM_PROMPTS_V2:
            raise KeyError(f"unknown maxim: {maxim}")
        return _TEMPLATE_V2.format(maxim_upper=maxim.upper(), instruction=instruction,
                                   context=context, **MAXIM_PROMPTS_V2[maxim])
    if maxim not in MAXIM_PROMPTS:
        raise KeyError(f"unknown maxim: {maxim}")
    spec = MAXIM_PROMPTS[maxim]
    return _TEMPLATE.format(maxim_upper=maxim.upper(), instruction=instruction,
                            context=context, **spec)


# --- baseline: LLM classifier with a direct question -----------------------
DIRECT_TEMPLATE = """Instruction: "{instruction}"
Environment: {context}

Is this instruction ambiguous for a robot in this environment?

Answer with a single word, YES or NO. Do not explain.
Answer:"""

DIRECT_SYSTEM = ("You classify robot instructions as ambiguous or unambiguous. "
                 "You answer with exactly one word: YES or NO.")


def direct_prompt(instruction: str, context: str) -> str:
    return DIRECT_TEMPLATE.format(instruction=instruction, context=context)


# --- baseline: LLM classifier with a maxim-structured prompt ---------------
STRUCTURED_TEMPLATE = """You judge whether a robot instruction is ambiguous, using Grice's
conversational maxims. An instruction is ambiguous if it violates at least one maxim:

- Quantity: too little or too much information to uniquely identify what to do.
- Quality: false, impossible, or incorrect assumptions about the world or context.
- Relation: not relevant to the current task or context.
- Manner: unclear, vague, or structurally ambiguous.

Instruction: "{instruction}"
Environment: {context}

Answer with a single word, YES (ambiguous) or NO (unambiguous). Do not explain.
Answer:"""


def structured_prompt(instruction: str, context: str) -> str:
    return STRUCTURED_TEMPLATE.format(instruction=instruction, context=context)
