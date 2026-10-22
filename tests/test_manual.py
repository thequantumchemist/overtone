"""
The manual (docs/manual.md) must document every parameter, field and public
method of the user-facing API, with the default values used in the code.
"""
import inspect
import re
from pathlib import Path

import pytest

import overtone
from overtone import spectrum
from overtone.anharmonics import transitions
from overtone.engines import SerialEngine
from overtone.harmonics.harmonics import is_linear_molecule

MANUAL = (Path(__file__).parents[1] / "docs" / "manual.md").read_text()
INTERNAL = {"serialize_array", "serialize_array_field", "serialize_dict", "validate_and_init"}

FUNCTIONS = [
    overtone.calculate_anharmonics, overtone.estimate_cost, overtone.recompute_vpt2,
    overtone.Harmonics, overtone.AnharmonicDerivativesSolver,
    overtone.AnharmonicDerivativesConfig.from_harmonics_result,
    overtone.VPT2Config.from_harmonics_result, overtone.DipoleDerivativesCalculator,
    overtone.VPT2Solver, overtone.TransitionCalculator, overtone.CoordinateTransformer,
    spectrum.plot_spectrum, spectrum.broaden_transitions, spectrum.broaden,
    spectrum.lorentzian, spectrum.gaussian, transitions.term_value,
    transitions.enumerate_states, is_linear_molecule,
]
MODELS = [overtone.HarmonicsResult, overtone.AnharmonicDerivativesConfig,
          overtone.AnharmonicDerivativesResult, overtone.DipoleDerivativesResult,
          overtone.VPT2Config, overtone.VPT2Result, overtone.VPT2Transition]
CLASSES = [overtone.Harmonics, overtone.HarmonicsResult, overtone.AnharmonicDerivativesSolver,
           overtone.AnharmonicDerivativesResult, overtone.DipoleDerivativesCalculator,
           overtone.VPT2Solver, overtone.VPT2Result, overtone.TransitionCalculator,
           overtone.CoordinateTransformer, SerialEngine]


BLOCKS = [re.sub(r"\s+", " ", b) for b in re.findall(r"```python\n(.*?)```", MANUAL, re.S)]


def signature_block(func):
    """The manual's code block that shows the signature of func."""
    name = func.__name__ if func.__name__ != "__init__" else func.__qualname__.split(".")[0]
    names = [p.name for p in params(func)]
    candidates = [b for b in BLOCKS if re.search(rf"\b{re.escape(name)}\(", b)]
    if not candidates:
        raise AssertionError(f"no signature block for {name} in the manual")
    # the signature block is the one that mentions most of the parameters
    return max(candidates, key=lambda b: sum(bool(re.search(rf"\b{n}\b", b)) for n in names))


def documented(name):
    return re.search(rf"\b{re.escape(name)}\b", MANUAL) is not None


def params(func):
    return [p for p in inspect.signature(func).parameters.values()
            if p.name not in ("self", "cls") and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.__qualname__)
def test_every_parameter_and_default_is_documented(func):
    block = signature_block(func)
    for p in params(func):
        assert documented(p.name), f"{func.__qualname__}: parameter {p.name} missing"
        if p.default is not inspect.Parameter.empty:
            text = f"{p.name}={p.default!r}".replace("'", '"')
            assert text in block, f"{func.__qualname__}: default '{text}' not in its signature block"


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_every_field_is_documented(model):
    for name in model.model_fields:
        assert documented(name), f"{model.__name__}: field {name} missing"


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_every_public_method_is_documented(cls):
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or name.startswith("model_") or name in INTERNAL:
            continue
        if name in {"construct", "copy", "dict", "json", "parse_file", "parse_obj",
                    "parse_raw", "schema", "schema_json", "update_forward_refs",
                    "validate", "from_orm"}:
            continue  # pydantic API
        if callable(member) or isinstance(inspect.getattr_static(cls, name), property):
            assert documented(name), f"{cls.__name__}.{name} missing"
