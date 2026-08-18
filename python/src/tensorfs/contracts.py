"""The predefined contract library, one constant per document.

``from tensorfs import contracts`` then
``lanes=(contracts.SDXL_DIFFUSERS_BF16, ...)``: the constant IS the layout —
a :class:`tensorfs.Contract` built at import from the packaged
``spec/v1/contracts`` documents, the same files the Rust core embeds (the
wheel carries them under ``tensorfs/_contracts``; in a source checkout they
are symlinks into ``spec/v1/contracts``, so there is no generated copy to
drift).

Constant names derive mechanically from contract names (``.``/``-`` → ``_``,
uppercase): ``sdxl.diffusers-bf16`` → ``SDXL_DIFFUSERS_BF16``. The
unversioned constant is the library's newest version; :func:`get` pins a
``name@version``; :func:`all` enumerates.
"""

from __future__ import annotations

from importlib.resources import files

from .contract import Contract

__all__ = ["all", "get"]

_builtin = tuple(  # noqa: RUF100 -- built once, at import, Rust-validated
    sorted(
        (
            Contract.from_document(entry.read_text(encoding="utf-8"))
            for entry in (files(__package__) / "_contracts").iterdir()
            if entry.name.endswith(".json")
        ),
        key=lambda contract: (contract.name or "", contract.version or 0),
    )
)

#: name -> newest contract; "name@version" -> that exact version.
_by_ref: dict[str, Contract] = {}
for _contract in _builtin:
    _by_ref[f"{_contract.name}@{_contract.version}"] = _contract
    _by_ref[str(_contract.name)] = _contract  # ascending sort: newest wins

_by_constant: dict[str, Contract] = {
    str(contract.name).replace(".", "_").replace("-", "_").upper(): contract
    for contract in _builtin
    if contract.version == _by_ref[str(contract.name)].version
}


def get(ref: str) -> Contract:
    """The library contract ``ref`` names: ``name`` (newest) or
    ``name@version`` (pinned)."""

    try:
        return _by_ref[ref]
    except KeyError:
        known = ", ".join(sorted(name for name in _by_ref if "@" in name))
        raise KeyError(f"no library contract named {ref!r}; the library holds: {known}") from None


def all() -> tuple[Contract, ...]:  # noqa: A001 -- the enumeration IS the point
    """Every packaged library contract, every version."""

    return _builtin


def __getattr__(name: str) -> Contract:
    """``contracts.SDXL_DIFFUSERS_BF16``-style constants, resolved against
    the packaged library. Unknown names refuse with the known set."""

    try:
        return _by_constant[name]
    except KeyError:
        known = ", ".join(sorted(_by_constant))
        raise AttributeError(
            f"module {__name__!r} has no contract constant {name!r}; it has: {known}"
        ) from None
