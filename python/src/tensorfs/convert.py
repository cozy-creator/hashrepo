"""Produce the checkpoint a v2 layout asks for, from the one you have.

The target is an :class:`~tensorfs.layout2.ExpectedHeader` — ``quant(topology)``
as the Go engine computed it. The verdict that a checkpoint is DERIVABLE to that
layout is the hub's and is taken in Go; this module is the other half, the thing
that can actually produce the derived checkpoint. Until it existed nothing could:
two independent gates emitted a CONVERTIBLE code and then refused, because the
producer they pointed at was never wired (tensorhub th#2164).

**The recipe is a property of the TARGET LAYOUT, never a caller's argument**
(torchcg tcg#53). :func:`recipe_for` reads it off ``target.quant``: the QUANT
RULE *is* the recipe — ``cozy.fp8-rowwise@1`` names one transformation and one
set of conventions, and there is nowhere else for that fact to live. A caller
that could pass its own recipe would be a second authority for a fact the layout
already states, which is exactly the serve-time cast tcg#53 deletes.

**No matching happens here (tensorfs#129, closed by deletion).** v1's planner
carried a port of the ``{i}``-hole pattern grammar so it could ask "which
declaration claims this tensor" — the THIRD copy of a rule whose authority was
Rust, and the copy #129 named. A v2 layout is a FINITE MAP, so the question is
``target.tensors(component).get(name)``, a dict lookup with nothing to disagree
about. The port, ``_claims``, ``_segments`` and ``_declaration_for`` are gone.

**What this module owns and what it deliberately does not.** tensorfs is
chunking and the local CAS. So it owns the PLAN — a pure function of (source
headers, computed layout) — and the WRITE, which is
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

from .layout2 import ExpectedHeader, LayoutTensor
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
    "component_of",
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

#: Recipe tokens: what the producer switches its kernel set on. They are the
#: PRODUCER's vocabulary, not the layout's — the layout names a quant rule, and
#: `_RECIPES` below is the one place the two are related.
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
    """The target layout's stamp, ``<topology>@v+<quant>@v``."""
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


# -- the recipe, which is the quant rule -------------------------------------
#
# v1 scanned the target's declarations for `F8_E4M3` beside `.weight_scale`
# twins and INFERRED a recipe from the shape of the document. v2 does not have
# to guess: `cozy.fp8-rowwise@1` names one transformation, and the conventions
# it carries are the producer's instructions. What replaced the inference is a
# lookup plus a refusal.

#: quant-rule family -> the kernel set this producer implements. A family that
#: is not here is a REFUSAL and never a default: silently falling through to a
#: cast for, say, `cozy.nvfp4-flat` writes 4-bit-shaped garbage — or, for an
#: fp8 family whose scales this producer cannot compute, fp8 bytes with no
#: dequant multiplier. Both files parse, load, and serve wrong numbers.
#:
#: `cozy.fp8-storage` is deliberately ABSENT. It is a real shipped rule, but a
#: scale-free one whose conventions require clamping to +/-448 before the cast
#: (torch's fp8 cast does not saturate, so an unclamped weight becomes NaN).
#: This producer has no clamp-then-cast kernel, and a plain cast would be the
#: exact silent failure the refusal exists for.
_RECIPES: Mapping[str, str] = {
    "plain.bf16": DTYPE_CAST,
    "plain.f16": DTYPE_CAST,
    "plain.f32": DTYPE_CAST,
    "cozy.fp8-rowwise": FP8_ROWWISE,
}


def recipe_for(target: ExpectedHeader) -> str:
    """Which conversion this layout's bytes are made by.

    Read off the target's quant rule, never passed in. The rule IS the recipe.
    """

    recipe = _RECIPES.get(target.quant.family)
    if recipe is None:
        known = ", ".join(sorted(_RECIPES))
        raise ConversionRefused(
            f"{target.stamp}: no kernel for quant rule {target.quant.handle}; "
            f"this producer implements {known}. Refusing to fall back to a "
            "cast, which would write bytes in the target's element type with "
            "none of the rule's scales"
        )
    if recipe == FP8_ROWWISE:
        _check_conventions(target)
    return recipe


def _check_conventions(target: ExpectedHeader) -> None:
    """The rowwise kernel's assumptions, asserted against the rule that asked
    for it. A rule may be re-versioned with a different scale granularity or a
    reciprocal convention while keeping its family; the kernel below would then
    write confidently wrong scales. So the conventions are read, not assumed."""

    conventions = target.quant.conventions
    expected = {"scale": "per_channel_out", "scale_dtype": SCALE_DTYPE, "amax_divisor": "448"}
    wrong = {
        key: conventions.get(key) for key, want in expected.items() if conventions.get(key) != want
    }
    if wrong:
        raise ConversionRefused(
            f"{target.quant.handle} states conventions this producer's rowwise "
            f"kernel does not implement: {wrong} (expected {expected})"
        )


# -- planning ---------------------------------------------------------------


def plan(
    reader: TensorReader,
    target: ExpectedHeader,
    *,
    component: str | None = None,
    strict: bool = True,
) -> ConversionPlan:
    """Decide, from headers alone, what the conversion will do.

    Reads no tensor data: every decision is a function of names, dtypes, shapes
    and the computed layout, which is the same property that makes a layout
    falsifiable from the header. So a plan can be computed on the control plane
    and shown to a human before a GPU is rented.

    ``component`` names which of the layout's finite maps these files are
    measured against; it may be omitted only when the layout has exactly one.
    """

    recipe = recipe_for(target)
    expected = target.tensors(component)
    files: list[FilePlan] = []
    for path in reader.files():
        views = [view for view in reader.values() if view.file == path]
        if not views:
            continue
        files.append(FilePlan(path, tuple(_plan_file(recipe, views, expected))))
    plan_ = ConversionPlan(recipe, target.stamp, tuple(files))
    if strict:
        _refuse_silent_failures(plan_, target)
    return plan_


def _plan_file(
    recipe: str, views: Sequence[TensorView], expected: Mapping[str, LayoutTensor]
) -> Iterable[Op]:
    present = {view.name for view in views}
    for view in sorted(views, key=lambda v: v.name):
        entry = expected.get(view.name)
        if entry is None:
            yield Op("inherit", view.name, "not a key of the target layout")
            continue
        if tuple(view.shape) != entry.shape:
            # v1 could only compare dtypes, so a checkpoint of the right family
            # and the wrong size converted cleanly and failed at load. The
            # computed layout carries the shape, so this is answerable now.
            raise ConversionRefused(
                f"{view.name}: the layout says {list(entry.shape)}, the "
                f"checkpoint says {list(view.shape)}; this is not that model"
            )
        if entry.accepts(view.dtype):
            yield Op("inherit", view.name, f"already {view.dtype}")
            continue
        wanted = entry.dtypes[0]
        if recipe == FP8_ROWWISE and wanted == FP8_E4M3:
            scale = view.name[: -len(".weight")] + SCALE_SUFFIX
            if scale not in expected:
                raise ConversionRefused(
                    f"{view.name}: the layout wants it {FP8_E4M3} and declares "
                    f"no {scale}; fp8 bytes with no dequant multiplier are not "
                    "a layout, they are a half-quantized artifact"
                )
            if scale in present:
                yield Op(
                    "inherit",
                    view.name,
                    f"already quantized — {scale} is present",
                )
                continue
            yield Op(
                "quantize-rowwise",
                view.name,
                f"{view.dtype} -> {FP8_E4M3} with a per-row {SCALE_DTYPE} scale",
                to_dtype=FP8_E4M3,
                emits_scale=scale,
            )
            continue
        yield Op("cast", view.name, f"{view.dtype} -> {wanted}", to_dtype=wanted)


def _refuse_silent_failures(plan_: ConversionPlan, target: ExpectedHeader) -> None:
    """The refusal that exists because the failure does not raise.

    tcg#53's: a conversion that RUNS and produces a file which is bit-identical
    to its source and stamped with a layout it does not implement. A crash
    would be better, because the pod serves this one.

    ``quant.transformed`` is the computed layout's own count of the keys the
    rule transformed — the number Go arrived at, not a second count taken from
    the declarations here.
    """

    if target.quant.transformed == 0:
        return
    quantized = [op for f in plan_.files for op in f.ops if op.kind == "quantize-rowwise"]
    # ALREADY-fp8 bytes are the other legitimate zero-conversion outcome, and
    # they arrive under two different reasons: the tensor's own dtype already
    # equals the layout's, or its scale twin is already present. Counting only
    # one of them turns a correct re-run into a refusal.
    resident = [
        op
        for f in plan_.files
        for op in f.ops
        if op.kind == "inherit"
        and (op.reason.startswith("already quantized") or op.reason == f"already {FP8_E4M3}")
    ]
    if quantized or resident:
        return
    raise ConversionRefused(
        f"{target.stamp} transforms {target.quant.transformed} tensor(s) and "
        "the plan converts none of them: the module shape moved under the rule, "
        "or this checkpoint is not that topology's model. Refusing to write a "
        "file that would be bit-identical to its source and stamped as "
        f"{target.quant.handle}"
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


def component_of(target: ExpectedHeader, path: str) -> str | None:
    """Which component of the layout a member belongs to, or ``None``.

    The multifolder packaging names its own components: `unet/...` is `unet`.
    This is v2's answer to the collision v1 could not resolve — SDXL's
    `text_encoder` and `text_encoder_2` both carry
    `text_model.encoder.layers.0.self_attn.q_proj.weight` at DIFFERENT shapes,
    and a single flat pattern set could not tell CLIP-L from CLIP-G. A finite
    map per component can.

    ``None`` is not a failure: a member the layout says nothing about (an
    unlisted folder, a scheduler config) is carried by reference untouched.
    """

    if len(target.components) == 1:
        return next(iter(target.components))
    head = path.split("/", 1)[0]
    return head if head in target.components else None


def convert(
    cas: LocalCAS,
    manifest: RepositoryManifest,
    target: ExpectedHeader,
    *,
    kernels: Mapping[str, Kernel] | None = None,
) -> ConversionResult:
    """The whole producer: a checkpoint tree in, a checkpoint tree out.

    **One reader per MEMBER, deliberately.** A diffusers multifolder tree
    legitimately repeats key spellings across component files, and
    :class:`~tensorfs.tensors.TensorReader` refuses a name that appears in two
    files because a single flat index cannot answer for it. The layout is per
    component anyway, so conversion is too, and the collision never arises. Do
    not "fix" this by flattening the tree into one reader.
    """

    plans: list[FilePlan] = []
    entries: dict[str, FileEntry] = {}
    recipe = recipe_for(target)
    for entry in manifest.files:
        component = component_of(target, entry.path)
        if component is None:
            plans.append(FilePlan(entry.path, ()))
            continue
        with open_tensors(cas, RepositoryManifest((entry,))) as reader:
            # strict=False: "the rule transformed nothing" is a property of the
            # CHECKPOINT, not of a member. An fp8 rule legitimately scopes
            # itself to the denoiser, so refusing per member would refuse every
            # correct conversion on its vae.
            member = plan(reader, target, component=component, strict=False)
            plans.extend(member.files)
            entries.update(apply(member, reader, cas, kernels=kernels))
    whole = ConversionPlan(recipe, target.stamp, tuple(plans))
    _refuse_silent_failures(whole, target)
    return ConversionResult(
        whole,
        RepositoryManifest(tuple(entries.get(entry.path, entry) for entry in manifest.files)),
        tuple(sorted(entries)),
    )
