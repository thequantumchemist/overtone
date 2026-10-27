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

MANUAL = (Path(__file__).parents[2] / "docs" / "manual.md").read_text()
INTERNAL = {"serialize_array", "serialize_array_field", "serialize_dict", "validate_and_init"}
PYDANTIC_API = {"construct", "copy", "dict", "json", "parse_file", "parse_obj", "parse_raw",
                "schema", "schema_json", "update_forward_refs", "validate", "from_orm"}

FUNCTIONS = [
    overtone.calculate_anharmonics, overtone.estimate_cost, overtone.recompute_vpt2,
    overtone.Harmonics, overtone.AnharmonicDerivativesSolver,
    overtone.AnharmonicDerivativesConfig.from_harmonics_result,
    overtone.VPT2Config.from_harmonics_result, overtone.DipoleDerivativesCalculator,
    overtone.VPT2Solver, overtone.TransitionCalculator, overtone.CoordinateTransformer,
    spectrum.plot_spectrum, spectrum.broaden_transitions, spectrum.broaden,
    spectrum.lorentzian, spectrum.gaussian, transitions.term_value,
    transitions.enumerate_states, is_linear_molecule, SerialEngine,
]
MODELS = [overtone.HarmonicsResult, overtone.AnharmonicDerivativesConfig,
          overtone.AnharmonicDerivativesResult, overtone.DipoleDerivativesResult,
          overtone.VPT2Config, overtone.VPT2Result, overtone.VPT2Transition]
CLASSES = [overtone.Harmonics, overtone.HarmonicsResult, overtone.AnharmonicDerivativesSolver,
           overtone.AnharmonicDerivativesResult, overtone.DipoleDerivativesCalculator,
           overtone.VPT2Solver, overtone.VPT2Result, overtone.TransitionCalculator,
           overtone.CoordinateTransformer, SerialEngine]


BLOCKS = [re.sub(r"\s+", " ", b) for b in re.findall(r"```python\n(.*?)```", MANUAL, re.S)]


def params(func):
    return [p for p in inspect.signature(func).parameters.values()
            if p.name not in ("self", "cls") and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]


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


def public_members(cls):
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or name.startswith("model_") or name in INTERNAL | PYDANTIC_API:
            continue
        if callable(member) or isinstance(inspect.getattr_static(cls, name), property):
            yield name


PARAMETERS = [pytest.param(f, p, id=f"{f.__qualname__}-{p.name}")
              for f in FUNCTIONS for p in params(f)]
DEFAULTS = [pytest.param(f, p, id=f"{f.__qualname__}-{p.name}")
            for f in FUNCTIONS for p in params(f) if p.default is not inspect.Parameter.empty]
FIELDS = [pytest.param(name, id=f"{m.__name__}-{name}") for m in MODELS for name in m.model_fields]
METHODS = [pytest.param(name, id=f"{c.__name__}-{name}") for c in CLASSES
           for name in public_members(c)]


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.__qualname__)
def test_signature_block_exists(func):
    signature_block(func)


@pytest.mark.parametrize("func, param", PARAMETERS)
def test_parameter_is_documented(func, param):
    assert documented(param.name)


@pytest.mark.parametrize("func, param", DEFAULTS)
def test_default_is_in_signature_block(func, param):
    text = f"{param.name}={param.default!r}".replace("'", '"')
    assert text in signature_block(func)


@pytest.mark.parametrize("name", FIELDS)
def test_field_is_documented(name):
    assert documented(name)


@pytest.mark.parametrize("name", METHODS)
def test_public_method_is_documented(name):
    assert documented(name)
