"""Produce the checkpoint a lane contract asks for, from the one you have.

This is the other half of the derivability verdict. ``Contract.Verdict`` (Go,
``verdict.go``) answers *Satisfies | DerivableVia(conversion) | Incompatible*
for a candidate checkpoint against a lane. The middle answer is only worth
having if something can actually produce the derived checkpoint, and until this
module existed nothing could: two independent gates emitted a CONVERTIBLE code
and then refused, because the producer they pointed at was never wired
(tensorhub th#2164 — ``grep "Conversions:"`` returned zero hits).

**The recipe is a property of the TARGET LANE, never a caller's argument**
(torchcg tcg#53). ``recipe_for`` reads it out of the target document: a lane
whose declarations name ``F8_E4M3`` beside per-row ``weight_scale`` twins IS an
fp8-rowwise lane, and there is nowhere else for that fact to live. A caller
that could pass its own recipe would be a second authority for a fact the lane
already states, which is exactly the serve-time cast tcg#53 deletes.

**What this module owns and what it deliberately does not.** tensorfs is
chunking and the local CAS. So it owns the PLAN — which is contract semantics,
a pure function of (source headers, target document) — and the WRITE, which is
:class:`~tensorfs.writer.TensorWriter` and was already here. It does NOT own
the numeric kernel. An op names what must happen; a ``kernels`` mapping supplies
the code that does it. :mod:`tensorfs.convert` ships a reference kernel set in
pure Python so the loop is runnable and falsifiable with no torch anywhere near
it, and production passes its own. That seam is why this module can be tested
end to end on a laptop and still be the thing a GPU pod runs.

**Nothing untouched is rewritten.** A file the plan does not touch is carried
by reference; inside a touched file every tensor the plan keeps is inherited by
its existing CAS objects. So converting an SDXL tree to the fp8-rowwise lane
re-admits the unet's block Linears and nothing else — the vae and both text
encoders keep every digest they had, and the hub has nothing new to fetch for
them.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .contract import Contract, TensorDecl
from .manifest import FileEntry, RepositoryManifest
from .tensors import TensorError, TensorReader, TensorView, open_tensors
from .writer import TensorWriter

if TYPE_CHECKING:
    from .local import LocalCAS

__all__ = [
    "ConversionPlan",
    "ConversionRefused",
    "ConversionResult",
    "FilePlan",
    "Op",
    "REFERENCE_KERNELS",
    "apply",
    "convert",
    "plan",
    "recipe_for",
]

#: The safetensors spelling of the only fp8 element type any shipped lane uses.
FP8_E4M3 = "F8_E4M3"
#: The dequant multiplier's spelling and element type. One scale per output row.
SCALE_SUFFIX = ".weight_scale"
SCALE_DTYPE = "F32"
#: torch's ``float8_e4m3fn`` cast does NOT saturate, so a producer must clamp
#: first or an out-of-range weight silently becomes NaN.
FP8_E4M3_MAX = 448.0

#: Recipe tokens. These are the same strings ``tensorfs.Conversion.Kind``
#: carries on the Go side, so a hub that read a verdict can name the recipe it
#: was offered without translating.
DTYPE_CAST = "dtype-cast"
FP8_ROWWISE = "fp8-rowwise"


class ConversionRefused(ValueError):
    """A conversion that must not be attempted, with the reason.

    Every refusal here exists because the corresponding FAILURE IS SILENT.
    A half-quantized lane serves plausible wrong numbers; a re-quantized
    artifact loses bits it can never get back; a recipe that matches nothing
    produces a file that is bit-identical to its source and stamps it with a
    lane it does not implement. None of those raise on their own.
    """


@dataclass(frozen=True)
class Op:
    """One tensor's fate, with the reason it was decided that way.

    ``reason`` is not decoration. The selector's refusals ARE the product
    (tcg#53): an operator looking at a converted checkpoint has to be able to
    ask "why is this Linear still bf16?" and get an answer that is not
    "someone's skip-list".
    """

    kind: str
    """``"inherit"``, ``"cast"`` or ``"quantize-rowwise"``."""
    name: str
    reason: str
    to_dtype: str = ""
    """The element type the op produces. Empty for ``inherit``."""
    emits_scale: str = ""
    """The scale tensor a ``quantize-rowwise`` op additionally writes."""


@dataclass(frozen=True)
class FilePlan:
    path: str
    ops: tuple[Op, ...]

    @property
    def touched(self) -> bool:
        return any(op.kind != "inherit" for op in self.ops)


@dataclass(frozen=True)
class ConversionPlan:
    """What a conversion will do, before it does any of it."""

    recipe: str
    target: str
    """The target lane's stamp."""
    files: tuple[FilePlan, ...] = field(default_factory=tuple)

    @property
    def converted(self) -> int:
        return sum(1 for f in self.files for op in f.ops if op.kind != "inherit")

    @property
    def touched_files(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files if f.touched)

    def kept(self) -> tuple[tuple[str, str], ...]:
        """``(tensor, reason)`` for everything the recipe did NOT convert."""

        return tuple(
            (op.name, op.reason) for f in self.files for op in f.ops if op.kind == "inherit"
        )

    def __str__(self) -> str:
        return (
            f"{self.recipe} to {self.target}: {self.converted} tensor(s) in "
            f"{len(self.touched_files)} of {len(self.files)} file(s)"
        )


# -- the pattern grammar ----------------------------------------------------
#
# A PORT, and labelled as one. `{i}` holes are the contract format's whole
# matching grammar (spec/v1/contracts/README.md: "No regex: matching is linear
# and unambiguous"), and the authority is `Contract::matches`
# (crates/tensorfs-core/src/contract.rs) with the Go port beside it in
# `match.go`. This is a THIRD implementation and that is a real cost, paid
# because the planner must know which declaration claims a tensor and the Rust
# matcher is not reachable from Python today. It is fenced by
# `test_conversion.py::test_python_claims_agree_with_the_go_matcher`, which
# drives the shipped library through both. Collapsing it into a pyo3 binding is
# the standing follow-up (tensorfs#128).


def _segments(pattern: str) -> tuple[str, ...] | None:
    """Split a pattern on its holes, or ``None`` if the pattern is ambiguous."""

    parts = pattern.split("{i}")
    if len(parts) == 1:
        return (pattern,)
    for part in parts[1:-1]:
        if not part:  # two adjacent holes
            return None
    for part in parts[1:]:
        if part[:1].isdigit():  # a digit immediately after a hole
            return None
    return tuple(parts)


def _claims(pattern: str, name: str) -> bool:
    """Does ``pattern`` describe the tensor called ``name``?"""

    segments = _segments(pattern)
    if segments is None:
        return False
    if len(segments) == 1:
        return name == segments[0]
    if not name.startswith(segments[0]):
        return False
    cursor = len(segments[0])
    for index, literal in enumerate(segments[1:], start=1):
        digits = cursor
        while digits < len(name) and name[digits].isdigit():
            digits += 1
        run = name[cursor:digits]
        if not run or (len(run) > 1 and run[0] == "0"):
            return False
        last = index == len(segments) - 1
        if last:
            return name[digits:] == literal
        if not name.startswith(literal, digits):
            return False
        cursor = digits + len(literal)
    return False


def _declaration_for(target: Contract, name: str) -> TensorDecl | None:
    """The FIRST declaration claiming ``name`` — first-declaration-wins, the
    same order the matcher applies, so the planner and the gate never disagree
    about which rule governs a tensor."""

    for decl in target.tensors:
        if _claims(decl.pattern, name):
            return decl
    return None


# -- the recipe, read out of the target lane --------------------------------


def recipe_for(target: Contract) -> str:
    """Which conversion this lane's bytes are made by.

    Read from the document, never passed in. Two lanes ship today:

    * a lane declaring ``F8_E4M3`` weights beside ``.weight_scale`` twins is
      **fp8-rowwise**;
    * any other lane whose declarations agree on one float element type is a
      **dtype-cast** target.

    A lane declaring fp8 weights and NO scale twins is refused rather than
    treated as a plain cast: fp8 bytes with no dequant multiplier are the
    half-quantized artifact, and inferring a recipe for it would produce
    exactly that.
    """

    dtypes = {d for decl in target.tensors for d in (decl.dtypes or ())}
    if FP8_E4M3 in dtypes:
        scales = [d for d in target.tensors if d.pattern.endswith(SCALE_SUFFIX)]
        if not scales:
            raise ConversionRefused(
                f"{target.stamp} declares {FP8_E4M3} weights but no "
                f"{SCALE_SUFFIX!r} twin: fp8 bytes with no dequant multiplier "
                "are not a layout, they are a half-quantized artifact"
            )
        return FP8_ROWWISE
    return DTYPE_CAST


def _cast_target(decl: TensorDecl | None) -> str:
    if decl is None or not decl.dtypes:
        return ""
    return decl.dtypes[0]


# -- planning ---------------------------------------------------------------


def plan(reader: TensorReader, target: Contract, *, strict: bool = True) -> ConversionPlan:
    """Decide, from headers alone, what the conversion will do.

    Reads no tensor data: every decision is a function of names, dtypes, ranks
    and the target document, which is the same property that makes a contract
    falsifiable from the header. So a plan can be computed on the control plane
    and shown to a human before a GPU is rented.
    """

    recipe = recipe_for(target)
    files: list[FilePlan] = []
    for path in reader.files():
        views = [view for view in reader.values() if view.file == path]
        if not views:
            continue
        files.append(FilePlan(path, tuple(_plan_file(recipe, views, target))))
    plan_ = ConversionPlan(recipe, target.stamp, tuple(files))
    if strict:
        _refuse_silent_failures(plan_, target)
    return plan_


def _plan_file(recipe: str, views: Sequence[TensorView], target: Contract) -> Iterable[Op]:
    present = {view.name for view in views}
    for view in sorted(views, key=lambda v: v.name):
        decl = _declaration_for(target, view.name)
        if decl is None:
            yield Op("inherit", view.name, "not claimed by the target lane")
            continue
        wanted = _cast_target(decl)
        if not wanted:
            yield Op("inherit", view.name, "the lane accepts any dtype here")
            continue
        if view.dtype == wanted:
            yield Op("inherit", view.name, f"already {wanted}")
            continue
        if recipe == FP8_ROWWISE and wanted == FP8_E4M3:
            scale = view.name[: -len(".weight")] + SCALE_SUFFIX
            if scale in present:
                yield Op(
                    "inherit",
                    view.name,
                    f"already quantized — {scale} is present",
                )
                continue
            if len(view.shape) != 2:
                raise ConversionRefused(
                    f"{view.name}: the lane declares it {FP8_E4M3} but it is "
                    f"rank {len(view.shape)}; a per-row scale needs an output "
                    "axis to be per-row over"
                )
            yield Op(
                "quantize-rowwise",
                view.name,
                f"{view.dtype} -> {FP8_E4M3} with a per-row {SCALE_DTYPE} scale",
                to_dtype=FP8_E4M3,
                emits_scale=scale,
            )
            continue
        yield Op("cast", view.name, f"{view.dtype} -> {wanted}", to_dtype=wanted)


def _refuse_silent_failures(plan_: ConversionPlan, target: Contract) -> None:
    """The two refusals that exist because the failure does not raise.

    Both are tcg#53's, and both are about a conversion that RUNS and produces a
    file — one that is bit-identical to its source, one that is missing the
    scales that make its bytes mean anything. A conversion that stamps such a
    file with a lane it does not implement is worse than a crash, because the
    pod then serves it.
    """

    if plan_.recipe != FP8_ROWWISE:
        return
    quantized = [op for f in plan_.files for op in f.ops if op.kind == "quantize-rowwise"]
    # ALREADY-fp8 bytes are the other legitimate zero-conversion outcome, and
    # they arrive under two different reasons: the tensor's own dtype already
    # equals the declaration's, or its scale twin is already present. Counting
    # only one of them turns a correct re-run into a refusal.
    resident = [
        op
        for f in plan_.files
        for op in f.ops
        if op.kind == "inherit"
        and (op.reason.startswith("already quantized") or op.reason == f"already {FP8_E4M3}")
    ]
    if quantized or resident:
        return
    declared = sum(1 for decl in target.tensors if FP8_E4M3 in (decl.dtypes or ()))
    raise ConversionRefused(
        f"{target.stamp} declares {declared} {FP8_E4M3} tensor(s) and the plan "
        "converts none of them: the module shape moved under the recipe, or "
        "this checkpoint is not this lane's model. Refusing to write a file "
        "that would be bit-identical to its source and stamped as fp8"
    )


# -- the reference kernels --------------------------------------------------
#
# Pure Python, no numpy, no torch. They exist so the loop is RUNNABLE and the
# ops have a falsifiable definition rather than a prose one; production passes
# its own (torch on a pod, where the same arithmetic runs on the card).

_FLOAT_READERS: dict[str, Callable[[bytes, int], float]] = {}


def _read_f32(buf: bytes, index: int) -> float:
    return float(struct.unpack_from("<f", buf, index * 4)[0])


def _read_f16(buf: bytes, index: int) -> float:
    return float(struct.unpack_from("<e", buf, index * 2)[0])


def _read_bf16(buf: bytes, index: int) -> float:
    lo = buf[index * 2]
    hi = buf[index * 2 + 1]
    return float(struct.unpack("<f", bytes((0, 0, lo, hi)))[0])


_FLOAT_READERS.update({"F32": _read_f32, "F16": _read_f16, "BF16": _read_bf16})


def _write_f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _write_f16(value: float) -> bytes:
    return struct.pack("<e", value)


def _write_bf16(value: float) -> bytes:
    # Round-to-nearest-even on the truncated 16 low bits, the same rule
    # torch.Tensor.bfloat16() applies. Truncation instead would bias every
    # magnitude downward, which is invisible in a diff and visible in a render.
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    lower = bits & 0xFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 if lower else bits >> 16
    return struct.pack("<H", rounded & 0xFFFF)


_FLOAT_WRITERS: dict[str, Callable[[float], bytes]] = {
    "F32": _write_f32,
    "F16": _write_f16,
    "BF16": _write_bf16,
}


def _encode_e4m3(value: float) -> int:
    """One float to one ``float8_e4m3fn`` byte, round-to-nearest-even.

    e4m3fn has no infinities: the all-ones exponent with a non-zero mantissa is
    NaN and 0xFF/0x7F are the only NaNs, so the largest finite magnitude is 448.
    A caller that does not clamp first gets NaN out of torch's own cast, which
    is why the quantize op scales into range BEFORE calling this.
    """

    if value != value:  # NaN
        return 0x7F
    sign = 0x80 if (value < 0 or (value == 0 and struct.pack("<f", value)[3] & 0x80)) else 0
    magnitude = abs(value)
    if magnitude >= 464.0:  # rounds above the 448 maximum
        return sign | 0x7E
    if magnitude == 0.0:
        return sign
    # Subnormals: exponent 0, step 2**-9.
    if magnitude < 2.0**-6:
        step = 2.0**-9
        quantum = magnitude / step
        unit = int(quantum)
        remainder = quantum - unit
        if remainder > 0.5 or (remainder == 0.5 and unit & 1):
            unit += 1
        return sign | min(unit, 0x07)
    exponent = 0
    scaled = magnitude
    while scaled >= 2.0:
        scaled /= 2.0
        exponent += 1
    while scaled < 1.0:
        scaled *= 2.0
        exponent -= 1
    mantissa = (scaled - 1.0) * 8.0
    unit = int(mantissa)
    remainder = mantissa - unit
    if remainder > 0.5 or (remainder == 0.5 and unit & 1):
        unit += 1
    if unit == 8:
        unit = 0
        exponent += 1
    if exponent > 8:
        return sign | 0x7E
    return sign | ((exponent + 7) << 3) | unit


#: What a kernel hands back for one output tensor: (dtype, shape, bytes).
Block = tuple[str, list[int], bytes]


def _kernel_cast(view: TensorView, op: Op) -> Block:
    reader = _FLOAT_READERS.get(view.dtype)
    writer = _FLOAT_WRITERS.get(op.to_dtype)
    if reader is None or writer is None:
        raise ConversionRefused(
            f"{view.name}: the reference kernel casts between "
            f"{sorted(_FLOAT_READERS)} only, not {view.dtype} -> {op.to_dtype}"
        )
    source = view.tobytes()
    count = 1
    for dim in view.shape:
        count *= dim
    out = bytearray()
    for index in range(count):
        out += writer(reader(source, index))
    return op.to_dtype, list(view.shape), bytes(out)


def _kernel_quantize_rowwise(view: TensorView, op: Op) -> tuple[Block, Block]:
    """``scale = amax(row)/448``; ``q = round(w/scale)``, clamped.

    The DEQUANT convention (multiply the fp8 value by the scale to recover the
    weight) is the one the serving loader reads. Storing the reciprocal instead
    is the classic silent bug here: it is five orders of magnitude out and
    every tensor still has the right name, dtype and shape.
    """

    reader = _FLOAT_READERS.get(view.dtype)
    if reader is None:
        raise ConversionRefused(f"{view.name}: cannot quantize from {view.dtype}")
    source = view.tobytes()
    rows, columns = int(view.shape[0]), int(view.shape[1])
    weights = bytearray()
    scales = bytearray()
    for row in range(rows):
        base = row * columns
        peak = 0.0
        for column in range(columns):
            magnitude = abs(reader(source, base + column))
            if magnitude > peak:
                peak = magnitude
        scale = max(peak / FP8_E4M3_MAX, 1e-12)
        scales += struct.pack("<f", scale)
        for column in range(columns):
            value = reader(source, base + column) / scale
            value = max(-FP8_E4M3_MAX, min(FP8_E4M3_MAX, value))
            weights.append(_encode_e4m3(value))
    return (
        (FP8_E4M3, [rows, columns], bytes(weights)),
        (SCALE_DTYPE, [rows], bytes(scales)),
    )


#: op kind -> kernel. Pass your own to :func:`apply` to run the same plan on a
#: card; the plan does not change. A ``cast`` kernel returns one
#: :data:`Block`; a ``quantize-rowwise`` kernel returns two, the weight and its
#: scale, because emitting the scale is not optional.
Kernel = Callable[[TensorView, Op], Any]

REFERENCE_KERNELS: Mapping[str, Kernel] = {
    "cast": _kernel_cast,
    "quantize-rowwise": _kernel_quantize_rowwise,
}


# -- applying ---------------------------------------------------------------


def apply(
    plan_: ConversionPlan,
    reader: TensorReader,
    cas: LocalCAS,
    *,
    kernels: Mapping[str, Kernel] | None = None,
) -> dict[str, FileEntry]:
    """Run the plan, returning one :class:`FileEntry` per TOUCHED file.

    Files the plan did not touch are absent from the result on purpose: there
    is nothing to write for them and the caller's existing entries are still
    correct. Returning them would invite a caller to re-admit bytes that never
    changed.
    """

    kernels = kernels or REFERENCE_KERNELS
    produced: dict[str, FileEntry] = {}
    for file_plan in plan_.files:
        if not file_plan.touched:
            continue
        writer = TensorWriter(cas, file_plan.path)
        for op in file_plan.ops:
            view = reader[op.name]
            if op.kind == "inherit":
                try:
                    writer.inherit(view)
                except TensorError:
                    # Not object-aligned under the source's grid, so it has no
                    # digest of its own to carry over. Re-admitting its bytes
                    # is correct and costs one small object.
                    writer.add(view.name, view.dtype, view.shape, view.tobytes())
                continue
            kernel = kernels.get(op.kind)
            if kernel is None:
                raise ConversionRefused(f"no kernel supplied for op {op.kind!r} on {op.name}")
            produced_blocks: tuple[Block, ...]
            names: tuple[str, ...]
            if op.kind == "quantize-rowwise":
                weight, scale = kernel(view, op)
                produced_blocks = (weight, scale)
                names = (op.name, op.emits_scale)
            else:
                produced_blocks = (kernel(view, op),)
                names = (op.name,)
            for tensor_name, (dtype, shape, data) in zip(names, produced_blocks, strict=True):
                writer.add(tensor_name, dtype, shape, data)
        produced[file_plan.path] = writer.finish()
    return produced


@dataclass(frozen=True)
class ConversionResult:
    plan: ConversionPlan
    manifest: RepositoryManifest
    """The converted tree. Members the plan did not touch are the SAME entries,
    so their chunk digests are unchanged and the hub has nothing to fetch."""
    rewritten: tuple[str, ...]


def convert(
    cas: LocalCAS,
    manifest: RepositoryManifest,
    target: Contract,
    *,
    kernels: Mapping[str, Kernel] | None = None,
) -> ConversionResult:
    """The whole producer: a checkpoint tree in, a checkpoint tree out.

    **One reader per MEMBER, deliberately.** A diffusers multifolder tree
    legitimately repeats key spellings across component files — SDXL's
    `text_encoder` and `text_encoder_2` both carry
    `text_model.encoder.layers.{i}.self_attn.q_proj.weight`, which is the same
    reason the lane contract cannot tell CLIP-L from CLIP-G — and
    :class:`~tensorfs.tensors.TensorReader` refuses a name that appears in two
    files, because a single flat index cannot answer for it. Matching is per
    component file anyway, so conversion is too, and the collision never arises.
    Do not "fix" this by flattening the tree into one reader.
    """

    plans: list[FilePlan] = []
    entries: dict[str, FileEntry] = {}
    recipe = ""
    for entry in manifest.files:
        with open_tensors(cas, RepositoryManifest((entry,))) as reader:
            # strict=False: "the recipe converted nothing" is a property of
            # the CHECKPOINT, not of a member. An fp8 lane legitimately touches
            # only the denoiser, so refusing per member would refuse every
            # correct conversion on its vae.
            member = plan(reader, target, strict=False)
            recipe = member.recipe
            plans.extend(member.files)
            entries.update(apply(member, reader, cas, kernels=kernels))
    whole = ConversionPlan(recipe, target.stamp, tuple(plans))
    _refuse_silent_failures(whole, target)
    return ConversionResult(
        whole,
        RepositoryManifest(tuple(entries.get(entry.path, entry) for entry in manifest.files)),
        tuple(sorted(entries)),
    )
