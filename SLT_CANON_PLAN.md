# SLT-Canon Implementation Plan

## Context

The formula-search pipeline (`experiments/train_formula_models.py` → `src/multirag/formula_search`) currently trains FastText embeddings on two Tangent-CFT tree representations: **SLT** (literal — `x` stays `V!x`, numbers stay `N!2`) and **SLT-TYPE** (fully erased — every identifier becomes bare `V`, every number becomes bare `N`). SLT-TYPE over-generalizes: it collapses `x+x` and `x+y` to the same `V+V` form, and loses numeric literals (`x²` → `V^N`), which likely hurts ranking precision on ARQMath Task 2.

**SLT-Canon** is a new representation that sits between them: canonical alpha-renaming of variables (first distinct variable → `x0`, second → `x1`, ...) that **preserves co-reference** (`x+x` → `x0+x0`, staying distinct from `x+y` → `x0+x1`) and **keeps numbers literal** (`x²` → `x0²`). The goal is to collapse only true naming-variance (the same formula written with different variable names) while retaining the structural information SLT-TYPE throws away.

This plan covers building `SLT-CANON` as a full new tree-type/representation usable everywhere `SLT`/`SLT-TYPE` are used today — CLI training, FAISS indexing (ScalarQuantizer family), and Task 2 query-time retrieval — plus the two experimental toggles the spec requires (strict/aggressive identifier classification, AC-normalization) and the validation tooling needed to sanity-check the transform before spending compute retraining on the full corpus.

Scope agreed with user: core transform + tests, full CLI/trainer/indexer wiring, a real (small, local) training run, AC-normalization, and the distinct-form-counts validation gate. **Explicitly excluded** (data not available in this checkout): qrels-based partition classification and the full-corpus nearest-neighbor sanity check — see "Out of scope" below.

---

## Key design decisions

These were derived from deep exploration of the existing pipeline (three research passes, one of which empirically ran the real LaTeXML→MathML→tree pipeline against the actual corpus) and verified firsthand against the source:

- **Hook point**: `src/multirag/formula_search/tuple_extraction.py::extract_tuples_from_mathml_direct()` (currently lines 26-76) is the single choke-point both corpus-indexing and query-time embedding already share — confirmed it's called identically from `experiments/task2_formula_retrieval.py::embed_query`, `experiments/build_collection_formula_index.py::embed_mathml`, and `src/multirag/indexing/formula/faiss_scalar_quantizer.py`. Canon hooks in right after `symbol_root = MathExtractor.convert_to_layoutsymbol(pmml)` and before `SymbolTree(symbol_root).get_pairs(...)`, mutating the already-built, freshly-constructed `LayoutSymbol` tree in place (each call builds a fresh tree — no shared/cached tree exists anywhere, so in-place mutation is safe). Wiring in here means training and query paths stay consistent automatically.
- **Why not reuse SLT-TYPE's erasure mechanism**: that erasure (`tuple_tokenizer.py::TupleTokenizationMode.Type`) runs per-tag, statelessly, inside `TupleTokenizer._tokenize_node` — it has no memory of other identifiers in the formula, so it structurally cannot express "same variable → same canonical name, first-seen order." Confirmed by reading `tuple_tokenizer.py` directly. Canon must run earlier, at the tree level.
- **Reading-order traversal needs its own child-priority order**, distinct from `LayoutSymbol.active_children()` (confirmed at `layout_symbol.py:174-196`, order `a,o,c,n,b,u,d,e,w`) — which must NOT be modified, since `get_pairs()` and other tree code depend on it. That order puts `next` *before* `below`, which would visit a baseline continuation before a subscript's contents — wrong for a case like `x_a + y` (identifier inside a subscript). Canon uses its own order with `next` strictly last (matches the spec's explicit recommendation).
- **Subscripts and decorations are already separate child nodes, not folded into tag text** — confirmed both by reading `layout_symbol.py` and empirically against the real corpus: `x_1` and `x` both produce tag `"V!x"`, with `1` living in a separate `.below` child; primes (`f''`) and accents (`\bar{x}`, `\hat{r}`) attach via `.above`/`.over` as separate `mo` children, base tag stays clean. This means Stage F ("rewrite only the renamed value") automatically satisfies "keep subscript/decoration untouched" — no extra stripping/reattaching logic needed for the dominant case. (One rare defensive case remains — see Stage D below.)
- **No existing "known function name" signal at the SLT layer** — `\sin(x)` parses to a plain `V!sin` node, indistinguishable from a real variable except by tag length. Classification is therefore keyed mostly on tag length (single Unicode codepoint = candidate variable; length > 1 = always fixed, which covers `sin/cos/log/lim/det/...` automatically with no hardcoded name list) plus small explicit sets for the genuinely ambiguous single-char cases (`f,g,h,e,i,j`) and reserved constants (`π`).
- **AC-normalization is real tree surgery, not a simple sort.** Empirically confirmed: precedence is *not* represented in the tree — `a + b*c` (no explicit parens) flattens into one single `.next` baseline chain with no grouping, so a naive "find runs of the same operator" scan can't distinguish `b*c` from two separate addends. It needs a precedence-aware two-pass approach (multiplicative runs resolved first, then additive runs over the resulting terms). Also, function application (`\sin(x)`) and implicit multiplication use two different invisible Unicode operators (U+2061, U+2062) that both get silently stripped during parsing (`math_symbol.py:125-126`), making `sin` immediately followed by `(x)` structurally identical to genuine implicit multiplication — this needs an explicit guard or `sin(x)` could be "reordered" to `(x)sin`.
- **`x−y` vs `y−x` collapsing is not an AC-normalization concern at all** — `"-"` is never in the commutative-operator set, so it's never reordered. The two forms still map to the same canonical pattern (`x0−x1` either way) purely because Stage E names by first-appearance position, blind to spelling. This is the spec's own documented, accepted residual (§9) — nothing to build for it.
- **Token-ID collision safety is achieved by construction, not by a reserved numeric block.** Canonical tags use a lexically-distinct sentinel format that no real parsed identifier could ever produce, so the existing dynamic `TokenIDManager.get_or_assign_id()` (`tuple_tokenizer.py:62-84`, first-seen assignment starting at node-id 60000) safely assigns them IDs with zero changes to that class.
- **Representation-string → tree-type mapping is duplicated across ~8 call sites** (`faiss_scalar_quantizer.py` alone has 3 copies; also `faiss_ivfflat.py`, `faiss_fused.py`, `task2_formula_retrieval.py`, `build_collection_formula_index.py`) all doing `{"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}.get(representation, "SLT")` — note the **silent fallback to plain SLT** for any unrecognized string. This plan consolidates the mapping into one function so a missed call site fails loudly (or at least consistently) instead of silently mis-training/mis-indexing as plain SLT.

---

## New module: `src/multirag/formula_search/slt_canon.py`

Pure tree-transform module, no I/O — fits alongside `layout_symbol.py`/`math_symbol.py`/`tuple_extraction.py`.

```python
CanonPolicy = Literal["strict", "aggressive"]

RESERVED_CONSTANTS: frozenset[str] = frozenset({"π"})
AMBIGUOUS_SINGLE_CHAR: frozenset[str] = frozenset({"f", "g", "h", "e", "i", "j"})
CANON_TAG_FORMAT = "V!#CANON{index}#"          # Stage H sentinel, see Naming below

# Stage C — reading order (own order; does NOT touch active_children())
READING_ORDER_SLOTS = ("within", "below", "above", "over", "under",
                       "pre_above", "pre_below", "element")   # "next" always last

def reading_order_children(node: LayoutSymbol) -> list[tuple[str, LayoutSymbol]]: ...
def walk_reading_order(root: LayoutSymbol | None): ...          # generator, pre-order

# Stage D — classification
def strip_combining_marks(value: str) -> tuple[str, str]: ...   # defensive fallback, see below
def classify_identifier(tag: str, policy: CanonPolicy) -> bool: ...  # True = renameable

# Stage E — canonical name assignment
def assign_canonical_names(root: LayoutSymbol | None, policy: CanonPolicy) -> dict[LayoutSymbol, str]: ...

# Stage F — rewrite
def rewrite_tree(assignment: dict[LayoutSymbol, str]) -> None: ...

# Stage B — AC-normalization (see dedicated section below)
def name_blind_key(node: LayoutSymbol, policy: CanonPolicy) -> tuple: ...
def ac_normalize_tree(root: LayoutSymbol | None, policy: CanonPolicy) -> LayoutSymbol | None: ...

# Entry point
def canonicalize_layout_tree(root, policy: CanonPolicy = "strict", ac_normalize: bool = False) -> LayoutSymbol | None:
    if root is None:
        return None
    if ac_normalize:
        root = ac_normalize_tree(root, policy)
    assignment = assign_canonical_names(root, policy)
    rewrite_tree(assignment)
    return root
```

**Classification logic** (Stage D) — matches the spec's strict/aggressive table exactly, with the length rule doing most of the work for free:
```
value = tag[2:]                       # strip "V!"
base, _decoration = strip_combining_marks(value)
if tag[0] == "?": return False        # wildcard, fixed by design
if base in RESERVED_CONSTANTS: return False
if len(base) != 1: return False       # multi-char -> function-like (sin, cos, log, ...), always fixed
if base in AMBIGUOUS_SINGLE_CHAR: return policy == "aggressive"
return True                           # ordinary single-letter Latin/Greek variable
```
`strip_combining_marks` is a **minor defensive fallback**, not the primary decoration mechanism — corpus grepping found ~10 rare cases per file where a combining diacritical mark is folded directly into `mi`/`mn` text (likely LaTeXML/OCR artifacts) rather than being a separate `.over`/`.above` child. Strip trailing Unicode category-Mn characters before the length check so these don't get misclassified as multi-char/fixed.

**Reading order rationale**: `within` first (read a bracketed/matrix group's contents before anything attached from outside), then `below`/`above` (subscript before superscript), then `over`/`under` (fraction numerator before denominator), then prescripts/`element`, with `next` always last. Verified against the spec's own example: for `x_a + y`, this order visits `x` → recurses into `below` (`a`) → then `next` (`+`, then `y`) → giving appearance order `[x, a, y]`, matching the spec's human-reading approximation. (Only "next last" is load-bearing per the spec; the relative order of the other 7 slots is a reasonable default.)

**Co-reference keying** (Stage E): the assignment dict is keyed by `base` (the bare letter after `strip_combining_marks`) — first time a given base is seen among renameable identifiers in reading order, assign the next canonical index; reuse it for repeats. Subscripts stay a separate untouched child, so `x_1`/`x_2` get the *same* canonical base tag but remain distinguishable via their distinct `N!1`/`N!2` subscript children.

**Rewrite** (Stage F): for each `(node, index)` in the assignment, `node.tag = CANON_TAG_FORMAT.format(index=index)`. Nothing else on the node or tree changes.

**Idempotency falls out for free**: `CANON_TAG_FORMAT` produces multi-character tag values (`"#CANON0#"`, len > 1), which `classify_identifier` already treats as fixed/non-renameable on any subsequent pass — so `canonicalize_layout_tree(canonicalize_layout_tree(tree))` is a no-op on the second call by construction, not by special-casing.

---

## AC-normalization (`ac_normalize_tree`)

Highest-complexity, highest-risk piece — built and tested in isolation (own milestone, own test class) before wiring into the CLI.

1. **Recurse bottom-up first**: normalize every non-`next` child (`within, below, above, over, under, pre_below, pre_above, element`) before touching this node's own baseline chain, so nested runs are already normalized when this level's `name_blind_key` is computed. `.element` (matrix cell-to-cell stepping) is recursed into but never itself treated as a reorderable run (cell order isn't commutative).
2. **Flatten the `.next` baseline chain** starting at this node into a plain list `[node0, node1, node2, ...]`.
3. **Pass 1 — multiplicative** (tighter-binding): scan the list and greedily group maximal runs of operand-shaped nodes connected by `MULTIPLICATIVE_OPERATORS` (`×` U+00D7, `⋅` U+22C5, `*`) or by bare adjacency (implicit multiplication, since the invisible-times operator U+2062 is stripped during parsing and leaves no trace). **Guard**: adjacency where the preceding node is a fixed multi-char identifier (`len(base) > 1`, e.g. `sin`, `log`, `lim`) is *never* treated as multiplication — this is what stops `\sin(x)` (also adjacency-with-nothing-between, via the similarly-stripped U+2061 function-application operator) from being misread as a product and reordered to `(x)sin`. Each qualifying run becomes one composite **Term**; runs of length 1 pass through unwrapped.
4. Within each multiplicative Term, stable-sort its operand nodes by `name_blind_key` (tie-break: original position), and recombine with the (interchangeable) connector tokens reused positionally.
5. **Pass 2 — additive** (looser-binding): scan the resulting Term-level list for maximal runs connected exclusively by `"+"`; stable-sort Terms within each run by `name_blind_key(term)`, same tie-break rule.
6. **Splice back**: rebuild `.next` pointers across the reordered sequence, preserving whatever originally preceded/followed the whole chain (only internal links within this run are rewritten).

`name_blind_key(node, policy)`: a recursive structural serialization — literal for numbers/operators/fixed identifiers/structure, but every *renameable* identifier collapses to one shared placeholder value, so the key doesn't care which variable is which, only shape. Recurses using `reading_order_children` (reuses Stage C's order) for determinism.

**Known, accepted residual** (documented in spec §9, not a bug to fix): fully symmetric children (e.g. `x + y`, two bare single-letter variables with nothing else attached) are name-blind-indistinguishable and won't be reordered — `test_stable_tie_break_symmetric_operands` should assert this stays unreordered rather than flag it as broken.

**First implementation step for this milestone**: a quick corpus grep to confirm/expand the exact operator glyph set in use (already found: `+`=U+002B, `-`=U+002D, `×`=U+00D7, `⋅`=U+22C5) before finalizing `ADDITIVE_OPERATORS`/`MULTIPLICATIVE_OPERATORS`.

---

## Integration changes

**`src/multirag/formula_search/tuple_extraction.py`** (currently `Literal["SLT","OPT","SLT-TYPE"]` at line 28): extend to include `"SLT-CANON"`; add `canon_policy: CanonPolicy = "strict"` and `ac_normalize: bool = False` params to `extract_tuples_from_mathml_direct`; call `canonicalize_layout_tree(symbol_root, canon_policy, ac_normalize)` right after `convert_to_layoutsymbol(pmml)` (line 64), gated on `tree_type == "SLT-CANON"`. Also add the consolidated mapping here (it's the natural single source of truth for tree_type semantics):
```python
REPRESENTATION_TREE_TYPE = {
    "slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE",
    "slt_canon": "SLT-CANON", "slt_canon_aggressive": "SLT-CANON", "slt_canon_ac": "SLT-CANON",
}
REPRESENTATION_CANON_KWARGS = {
    "slt_canon": {"canon_policy": "strict", "ac_normalize": False},
    "slt_canon_aggressive": {"canon_policy": "aggressive", "ac_normalize": False},
    "slt_canon_ac": {"canon_policy": "strict", "ac_normalize": True},
}
def resolve_representation(representation: str) -> tuple[str, dict]:
    return REPRESENTATION_TREE_TYPE.get(representation, "SLT"), REPRESENTATION_CANON_KWARGS.get(representation, {})
```
`extract_tuples_from_latex_subprocess` (the LaTeX-subprocess mirror, used only by the excluded IVFFlat indexer and the legacy non-default `FormulaTrainer`) gets a `# TODO(slt-canon)` note but is **not** wired — see Out of scope.

**`src/multirag/embedding/formula_trainer_direct.py`** (`FormulaTrainerDirect`, the live trainer): extend the validated `tree_type` set (currently `{"SLT","OPT","SLT-TYPE"}` at line 314) to add `"SLT-CANON"`; add `canon_policy: Optional[Literal["strict","aggressive"]] = None` and `ac_normalize: Optional[bool] = None` constructor params, following the exact `Optional[...] = None, auto-unless-overridden` pattern already used for `embedding_type`/`tokenize_number` (lines 320-344). **Required fix**: the `tokenize_number` auto-derivation at line 342 (`self.tree_type == "SLT"`) must become `self.tree_type in ("SLT", "SLT-CANON")` — Canon keeps literal numbers like SLT, unlike SLT-TYPE, so it needs the same default (otherwise Canon would silently erase numbers). The `embedding_type` auto-derivation (lines 320-327) needs no change — its `else` branch already correctly yields `Both_Separated` for any non-SLT-TYPE tree_type. Thread `canon_policy`/`ac_normalize` into the `extract_tuples_from_mathml_direct(...)` call in `process_latex_formula` (line 550) and into metadata persistence.

**`src/multirag/embedding/formula_model_manager.py`**: add `canon_policy`/`ac_normalize` params to `train_from_corpus_file` and `_save_metadata`, persisted into the metadata JSON dict alongside the existing `tree_type`/`embedding_type`/`tokenize_number` fields — this is the exact mechanism `faiss_scalar_quantizer.py`'s query/index-time setup already uses to recover `embedding_type`/`tokenize_number`; Canon's settings must round-trip the same way or query-time canonicalization could silently diverge from training-time settings.

**`experiments/train_formula_models.py`**: extend `-t/--tree-type` `choices` (line 81) to add `"SLT-CANON"`; add `--canon-policy` (`choices=["strict","aggressive"], default="strict"`) and `--ac-normalize` (`action="store_true"`, the first boolean flag in this script); pass both through to the `FormulaTrainerDirect(...)` constructor call (lines 164-171).

**`src/multirag/formula_search/__init__.py`**: export `canonicalize_layout_tree`, `classify_identifier`, `resolve_representation` alongside the existing export block.

**Indexer/query-side wiring** — replace the local `tree_type_mapping` dict with an import of `resolve_representation` from `tuple_extraction.py`, and recover `canon_policy`/`ac_normalize` from training metadata the same way `embedding_type`/`tokenize_number` are recovered today, at each of:
- `src/multirag/indexing/formula/faiss_scalar_quantizer.py` (3 copies of the dict to replace)
- `experiments/task2_formula_retrieval.py` (`load_model_and_tokenizer`, and the `embed_query` call site)
- `experiments/build_collection_formula_index.py` (`setup_model_and_tokenizer`, and the `embed_mathml` call site)

**`src/multirag/config/run_config.py`**: extend the representation validator sets in `RerankerConfig` and `Task2Config` (currently `{"slt","opt","slt_type"}`) to add `"slt_canon"`, `"slt_canon_aggressive"`, `"slt_canon_ac"`.

**New Task 2 condition configs**, mirroring the existing flat shape of `configs/task2/v1_slt.yaml` exactly (only `representation` differs):
- `configs/task2/v1_slt_canon.yaml` (`representation: "slt_canon"`)
- `configs/task2/v1_slt_canon_aggressive.yaml` (`representation: "slt_canon_aggressive"`)
- `configs/task2/v1_slt_canon_ac.yaml` (`representation: "slt_canon_ac"`)

---

## Validation: distinct-form-counts gate

New script `experiments/canon_distinct_forms.py` (argparse CLI, following the existing `experiments/a3_summarize.py`-style analysis-script convention — not the unittest convention, since this produces a report, not pass/fail assertions). Reads formulas from the locally-available `data/raw/collection/formula/slt_representation_v3/*.tsv` (reuse `parse_file_numbers`-style file selection from `train_formula_models.py`), and for a sample of formulas computes tuple signatures under SLT, SLT-TYPE, `slt_canon`, and `slt_canon_aggressive` via `extract_tuples_from_mathml_direct`. Reports `len(set(signatures))` per representation and flags the pre-committed decision rule: `distinct(SLT) > distinct(Canon) >> distinct(Type)` expected; `Canon ≈ SLT` means renaming is inert; `Canon ≈ Type` means classification is over-collapsing. This is runnable now, locally, with no missing-data blockers, and should be run before investing in a full retrain.

---

## Milestone sequence

1. **Core module** (`slt_canon.py`): traversal, classification, naming, rewrite. Exercise both strict and aggressive via the existing `policy` param — no separate code path needed.
2. **Unit tests** (`test/test_slt_canon.py`, stdlib `unittest`, mirroring `test/test_c0_3_reliability.py`'s conventions — see Test plan below).
3. **Wire into `tuple_extraction.py`** (new branch + `resolve_representation` helper). Sanity-check via `encode_tuples`/`TokenIDManager` that the `#CANON{i}#` sentinel tags get stable, collision-free IDs.
4. **CLI/trainer/metadata wiring** (`train_formula_models.py`, `formula_trainer_direct.py`, `formula_model_manager.py`). Run one small real local training job against the existing `slt_representation_v3` TSVs to prove the pipeline produces a valid FastText model + metadata JSON + encoder-maps TSV with `canon_policy`/`ac_normalize` correctly persisted.
5. **Distinct-form-counts script** (`experiments/canon_distinct_forms.py`) — run against local data as the go/no-go sanity check.
6. **AC-normalization** (`ac_normalize_tree`, `name_blind_key`) — corpus-grep to confirm operator glyphs first, then build with its own dedicated test class (mixed-precedence, function-application guard, idempotency), fully isolated before wiring `--ac-normalize` into the CLI.
7. **Indexer/query-side wiring** (`faiss_scalar_quantizer.py`, `task2_formula_retrieval.py`, `build_collection_formula_index.py`, `run_config.py` validators) + new Task2 YAML configs. Smoke-test using `Task2Config.index_limit` (already exists "for smoke tests" per its own docstring) — verifies no crashes and non-degenerate embeddings even without real qrels.

---

## Test plan (`test/test_slt_canon.py`)

Follows `test/test_c0_3_reliability.py`'s exact convention: stdlib `unittest.TestCase`, `sys.path.insert(...)` for `src/` at the top of the file, hand-built fixtures in `setUp()`, no pytest.

**Determinism contract:**
- `test_traversal_order_independent_of_identifier_values` — two structurally-identical trees differing only in which letters are used; assert identical canonical-tag *pattern*.
- `test_idempotency` — `canonicalize_layout_tree` applied twice (via `LayoutSymbol.Copy()` for an independent copy) produces byte-identical tuples to applying it once.
- `test_stable_tie_break_symmetric_operands` — `x + y` under `ac_normalize=True` stays unreordered (documented residual, not a bug).
- `test_repeated_runs_byte_identical` — two independent fresh parses of the same MathML produce identical output; guards against any accidental shared/module-level state in `assign_canonical_names`.

**Classification/rewrite correctness:**
- `test_x_plus_x_same_canonical_name`, `test_x_plus_y_different_canonical_names`
- `test_x_squared_number_preserved` — `N!2` child byte-identical after rename
- `test_subscript_distinguishability` — `x_1` vs `x_2`: same canonical base, subscript children untouched and still distinct
- `test_strict_vs_aggressive_divergence_f` / `_e` / `_i` / `_j`
- `test_pi_always_fixed_both_policies`
- `test_multichar_function_name_always_fixed` (e.g. `sin`)
- `test_decorated_identifier_hat` / `_prime`
- `test_empty_formula_identity_passthrough` — `canonicalize_layout_tree(None)` doesn't crash

**AC-normalization (separate test class):**
- `test_ac_normalize_pure_additive_chain`, `test_ac_normalize_implicit_multiplicative_chain` (non-symmetric operands, so reordering is checkable)
- `test_ac_normalize_mixed_precedence` — the `a + b*c` case
- `test_ac_normalize_function_application_guard` — `\sin(x) + 1`-shaped tree; confirm `sin`/`(x)` adjacency is never reordered
- `test_ac_normalize_idempotent`
- `test_ac_normalize_runs_before_renaming` — confirm canonical numbering reflects post-reorder order

---

## Verification (how to test end-to-end)

1. `python -m unittest test/test_slt_canon.py` — all pass.
2. Hand-check a few constructed examples through `extract_tuples_from_mathml_direct(mathml, tree_type="SLT-CANON")`: confirm `x+x` gives matching tags where `x+y` doesn't, and `x^2`'s `N!2` survives.
3. Small real training run: `python experiments/train_formula_models.py -t SLT-CANON -f "<one local file number>" -n 500 --canon-policy strict`. Confirm it completes and produces `training_metadata_slt_canon.json` with `canon_policy`/`ac_normalize` recorded correctly.
4. `python experiments/canon_distinct_forms.py` against the same local file(s); inspect reported counts against the decision rule.
5. Repeat 3 with `--ac-normalize` and `--canon-policy aggressive` to confirm both toggles run cleanly.
6. Build a small FAISS index via `experiments/build_collection_formula_index.py` with `representation=slt_canon` and a small `Task2Config.index_limit`; confirm no crashes, non-degenerate vectors.
7. Run `experiments/run_task2_experiment.py` with one of the new YAML configs; confirm the retrieval path runs end-to-end (scoring will be degenerate given missing qrels, but retrieval itself should execute cleanly).

---

## Out of scope (documented, not built)

- **`faiss_ivfflat.py`** — separate code path (uses `extract_tuples_from_latex_subprocess`), ~31.5GB memory footprint per representation per its own docstring. Not extended.
- **`faiss_fused.py`** — fixed 3-way sum architecture (`_REPRESENTATIONS = ["slt","opt","slt_type"]`), not a natural fit for a 4th channel, and not one of the spec's 6 named conditions. Not extended.
- **`extract_tuples_from_latex_subprocess`** — left with a `# TODO` note, not wired (only used by the two excluded paths above).
- **§7(b) partition classification over Task 2 qrels** — qrels/topics don't exist in this local checkout (only Task 1 data is present; real data lives in a GCS bucket per `notebooks/train_fasttext_in_vm.sh`). The classifier function's *design* (`classify_pair(query_sig, candidate_sig) -> "exact"|"alpha_variant"|"structural_only"|"semantic"`) is straightforward from what's already built, but not implemented/run in this pass.
- **§7(d) nearest-neighbor sanity check** — needs a full-corpus-trained model to be meaningful; the small local run in the milestone list is a pipeline smoke test, not a real embedding model.
- **Condition 6 (OPT constant-folding)** — spec marks this optional/later; not attempted.
