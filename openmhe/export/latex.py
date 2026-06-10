"""Render an :class:`~openmhe.ObjectiveBuilder` as a LaTeX MHE problem.

The output is the full moving-horizon estimation problem: an ``argmin`` over the
window variables, horizon summations of the weighted error terms, and an
arrival-cost term when an :class:`~openmhe.BaseArrivalCost` is supplied.
Set ``underbrace=True`` to label each term.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _term_kind(term) -> str:
    """Return ``target_type`` (or legacy ``type``) for a builder term."""
    return getattr(term, "target_type", getattr(term, "type", ""))


@dataclass
class LatexSymbols:
    """Configurable LaTeX symbols used when rendering an objective.

    Defaults follow the convention where the stored ``NoiseWeight.W`` is an
    inverse covariance, so weighted norms carry inverse-covariance subscripts
    (``R^{-1}``, ``Q^{-1}``, ``U^{-1}``, ``P^{-1}``).
    """

    output: str = "y"
    state: str = "x"
    input: str = "u"
    process_noise: str = "w"
    meas_noise: str = "v"
    defect: str = r"\delta"
    input_error: str = "q"
    A: str = "A"
    B: str = "B"
    C: str = "C"
    D: str = "D"
    cov_meas: str = "R"
    cov_proc: str = "Q"
    cov_input: str = "U"
    arrival_cov: str = "P"
    arrival_ref: str = r"\bar{x}"
    reg_weight: str = r"\lambda"
    horizon: str = "N"
    time: str = "t"
    index: str = "k"
    norm_open: str = r"\left\lVert "
    norm_close: str = r"\right\rVert"
    descriptions: dict = field(
        default_factory=lambda: {
            "arrival": "arrival cost",
            "MEASUREMENT": "measurement error",
            "PROCESS": "process noise",
            "INPUT_TRACKING": "input error",
            "FIRST_DIFF": "input rate",
            "SECOND_DIFF": "input acceleration",
            "INPUT_RANDOM_WALK": "random walk",
        }
    )


def _inv(symbol: str) -> str:
    """LaTeX inverse-covariance superscript."""
    return f"{symbol}^{{-1}}"


def _weighted_norm(residual: str, weight: str, sym: LatexSymbols, squared: bool = True) -> str:
    """Weighted norm ``||r||_W`` (optionally squared) in LaTeX."""
    sq = "^2" if squared else ""
    return f"{sym.norm_open}{residual} {sym.norm_close}{sq}_{{{weight}}}"


def _penalty_kind(term) -> str:
    """Name of the outer penalty class on a term."""
    penalty = getattr(term, "penalty", None)
    return type(penalty).__name__ if penalty is not None else "L2Penalty"


def _render_penalty(residual: str, weight: str, term, sym: LatexSymbols) -> str:
    """Render one error block according to its penalty type."""
    kind = _penalty_kind(term)
    if kind == "L1Penalty":
        return f"{sym.norm_open}{residual} {sym.norm_close}_{{{weight}, 1}}"
    if kind == "HuberPenalty":
        delta = getattr(term.penalty, "delta", 1.0)
        return (
            rf"\sum_i {weight}_{{ii}}\, \mathcal{{H}}_{{{_fmt_num(delta)}}}"
            rf"\left( [{residual}]_i \right)"
        )
    if kind == "DeadzonePenalty":
        return (
            rf"\sum_i {weight}_{{ii}}\, \mathcal{{D}}_{{z}}"
            rf"\left( [{residual}]_i \right)"
        )
    return _weighted_norm(residual, weight, sym, squared=True)


def _fmt_num(value: float) -> str:
    """Compact numeric literal for LaTeX."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _idx_super(target_idx) -> str:
    """Superscript for one or more input indices, e.g. ``(0)`` or ``(0,1)``."""
    idx = list(target_idx)
    if len(idx) == 1:
        return f"({idx[0]})"
    return "(" + ",".join(str(i) for i in idx) + ")"


def _reference_latex(reference, sym: LatexSymbols, sup: str) -> str:
    """LaTeX for an input-tracking reference at time ``k``."""
    k = sym.index
    if reference == "measured":
        return f"{sym.input}^{{{sup},\\mathrm{{ref}}}}_{{{k}}}"
    if reference == "zero":
        return "0"
    return _fmt_num(float(reference))


def _wrap_underbrace(body: str, description: str | None) -> str:
    """Optionally wrap ``body`` in ``\\underbrace{...}_{\\text{...}}``."""
    if description is None:
        return body
    return rf"\underbrace{{{body}}}_{{\text{{{description}}}}}"


def _meas_window(sym: LatexSymbols) -> tuple[str, str]:
    """Summation bounds for measurement terms over the horizon."""
    return f"{sym.time}-{sym.horizon}", f"{sym.time}"


def _proc_window(sym: LatexSymbols) -> tuple[str, str]:
    """Summation bounds for process / input terms (lags by one step)."""
    return f"{sym.time}-{sym.horizon}", f"{sym.time}-1"


def _sum_prefix(lo: str, hi: str, sym: LatexSymbols) -> str:
    """Horizon sum prefix ``\\sum_{k=lo}^{hi}``."""
    return rf"\sum_{{{sym.index}={lo}}}^{{{hi}}} "


def _render_arrival_term(
    arrival_cost,
    sym: LatexSymbols,
    underbrace: bool,
) -> str:
    """Render the arrival-cost block for a configured strategy."""
    lo, _ = _meas_window(sym)
    residual, weight, description = arrival_cost.latex_parts(
        state=sym.state, window_lo=lo
    )
    body = _weighted_norm(residual, weight, sym, squared=True)
    desc = description if underbrace else None
    return _wrap_underbrace(body, desc)


def objective_to_latex(
    builder,
    *,
    underbrace: bool = False,
    multiline: bool | None = None,
    environment: str = "equation",
    standalone: bool = False,
    symbols: LatexSymbols | None = None,
    define_penalties: bool = False,
    form: str = "substituted",
) -> str:
    """Render ``builder`` as a LaTeX MHE optimization problem.

    Parameters
    ----------
    builder : ObjectiveBuilder
        Objective whose terms are rendered, in order. When
        ``builder.arrival_cost`` is set, the corresponding arrival term is
        included automatically.
    underbrace : bool
        Label each term with an ``\\underbrace{...}_{\\text{...}}`` description.
    multiline : bool, optional
        Use an ``aligned`` block (one term per line). Defaults to ``underbrace``.
    environment : str
        Outer math environment (``"equation"``, ``"align"``, or ``""`` for none).
    standalone : bool
        Wrap the result in a minimal compilable LaTeX document.
    symbols : LatexSymbols, optional
        Symbol overrides.
    define_penalties : bool
        Prepend one-line definitions of any non-L2 penalty functions used.
    form : str
        ``"substituted"`` (default) writes the cost with residuals inlined.
        ``"constrained"`` (aliases ``"subject_to"``) writes a ``minimize`` over
        decision variables with the dynamics, measurement, and difference
        defects listed under ``subject to``.
    """
    sym = symbols or LatexSymbols()
    arrival_cost = getattr(builder, "arrival_cost", None)
    if form.lower() in ("constrained", "subject_to", "subject-to"):
        result = _constrained_latex(
            builder,
            sym=sym,
            underbrace=underbrace,
            arrival_cost=arrival_cost,
            environment=environment,
            define_penalties=define_penalties,
        )
        return _wrap_standalone(result) if standalone else result
    if multiline is None:
        multiline = underbrace

    k = sym.index
    terms = list(builder.terms)

    lines: list[str] = []

    if arrival_cost is not None:
        lines.append(_render_arrival_term(arrival_cost, sym, underbrace))

    has_process = any(_term_kind(t) == "PROCESS" for t in terms)
    has_input = any(
        _term_kind(t) in ("INPUT_TRACKING", "INPUT_REG", "INPUT_RANDOM_WALK")
        for t in terms
    )

    for term in terms:
        kind = _term_kind(term)
        if kind == "MEASUREMENT":
            lo, hi = _meas_window(sym)
            residual = (
                f"{sym.output}_{{{k}}} - "
                f"({sym.C} {sym.state}_{{{k}}} + {sym.D} {sym.input}_{{{k}}})"
            )
            body = _render_penalty(residual, _inv(sym.cov_meas), term, sym)
            desc = sym.descriptions[kind] if underbrace else None
            lines.append(_sum_prefix(lo, hi, sym) + _wrap_underbrace(body, desc))
        elif kind == "PROCESS":
            lo, hi = _proc_window(sym)
            residual = f"{sym.process_noise}_{{{k}}}"
            body = _render_penalty(residual, _inv(sym.cov_proc), term, sym)
            desc = sym.descriptions[kind] if underbrace else None
            lines.append(_sum_prefix(lo, hi, sym) + _wrap_underbrace(body, desc))
        elif kind == "INPUT_TRACKING":
            lo, hi = _proc_window(sym)
            sup = _idx_super(term.target_idx)
            ref = _reference_latex(getattr(term, "reference", "measured"), sym, sup)
            residual = f"{sym.input}^{{{sup}}}_{{{k}}} - {ref}"
            body = _render_penalty(residual, _inv(sym.cov_input), term, sym)
            desc = sym.descriptions[kind] if underbrace else None
            lines.append(_sum_prefix(lo, hi, sym) + _wrap_underbrace(body, desc))
        elif kind == "INPUT_REG":
            lo, hi = _proc_window(sym)
            sup = _idx_super(term.target_idx)
            trend = str(getattr(term, "trend", "FIRST_DIFF")).upper()
            if trend in ("FIRST_DIFF", "FIRST"):
                residual = (
                    f"{sym.input}^{{{sup}}}_{{{k}}} - "
                    f"{sym.input}^{{{sup}}}_{{{k}-1}}"
                )
                desc_key = "FIRST_DIFF"
            else:
                residual = (
                    f"{sym.input}^{{{sup}}}_{{{k}}} - "
                    f"2 {sym.input}^{{{sup}}}_{{{k}-1}} + "
                    f"{sym.input}^{{{sup}}}_{{{k}-2}}"
                )
                desc_key = "SECOND_DIFF"
            norm = _weighted_norm(residual, "", sym, squared=True).replace("_{}", "")
            body = f"{sym.reg_weight}\\, {norm}"
            desc = sym.descriptions[desc_key] if underbrace else None
            lines.append(_sum_prefix(lo, hi, sym) + _wrap_underbrace(body, desc))
        elif kind == "INPUT_RANDOM_WALK":
            lo, hi = _proc_window(sym)
            sup = _idx_super(term.target_idx)
            residual = (
                f"{sym.input}^{{{sup}}}_{{{k}}} - "
                f"{sym.input}^{{{sup}}}_{{{k}-1}}"
            )
            norm = _weighted_norm(residual, "", sym, squared=True).replace("_{}", "")
            body = f"{sym.reg_weight}\\, {norm}"
            desc = sym.descriptions[kind] if underbrace else None
            lines.append(_sum_prefix(lo, hi, sym) + _wrap_underbrace(body, desc))
        else:
            raise ValueError(f"Cannot render unknown term type: {kind!r}")

    min_vars = _min_subscript(sym, has_process, has_input)
    body_tex = _assemble(lines, min_vars, multiline)
    result = _wrap_environment(body_tex, environment, multiline)

    if define_penalties:
        preamble = _penalty_definitions(terms, sym)
        if preamble:
            result = preamble + "\n" + result

    if standalone:
        result = _wrap_standalone(result)
    return result


def _constrained_latex(
    builder,
    *,
    sym: LatexSymbols,
    underbrace: bool,
    arrival_cost,
    environment: str,
    define_penalties: bool,
) -> str:
    """Render the MHE problem in ``minimize ... subject to ...`` form."""
    terms = list(builder.terms)
    k = sym.index

    reg_terms = [
        t for t in terms if _term_kind(t) in ("INPUT_REG", "INPUT_RANDOM_WALK")
    ]
    single_defect = len(reg_terms) == 1 and len(
        list(getattr(reg_terms[0], "target_idx", [0]))
    ) == 1

    def _defect_sym(term) -> str:
        """LaTeX symbol for a regulator defect variable at time ``k``."""
        if single_defect:
            return f"{sym.defect}_{{{k}}}"
        return f"{sym.defect}^{{{_idx_super(term.target_idx)}}}_{{{k}}}"

    has_process = any(_term_kind(t) == "PROCESS" for t in terms)
    has_meas = any(_term_kind(t) == "MEASUREMENT" for t in terms)

    obj_parts: list[str] = []
    defect_constraints: list[str] = []

    if arrival_cost is not None:
        obj_parts.append(_render_arrival_term(arrival_cost, sym, underbrace))

    for term in terms:
        kind = _term_kind(term)
        if kind == "MEASUREMENT":
            lo, hi = _meas_window(sym)
            var = f"{sym.meas_noise}_{{{k}}}"
            body = _sum_prefix(lo, hi, sym) + _render_penalty(
                var, _inv(sym.cov_meas), term, sym
            )
            obj_parts.append(
                _wrap_underbrace(body, sym.descriptions[kind] if underbrace else None)
            )
        elif kind == "PROCESS":
            lo, hi = _proc_window(sym)
            var = f"{sym.process_noise}_{{{k}}}"
            body = _sum_prefix(lo, hi, sym) + _render_penalty(
                var, _inv(sym.cov_proc), term, sym
            )
            obj_parts.append(
                _wrap_underbrace(body, sym.descriptions[kind] if underbrace else None)
            )
        elif kind == "INPUT_TRACKING":
            lo, hi = _proc_window(sym)
            sup = _idx_super(term.target_idx)
            var = f"{sym.input_error}^{{{sup}}}_{{u,{k}}}"
            body = _sum_prefix(lo, hi, sym) + _render_penalty(
                var, _inv(sym.cov_input), term, sym
            )
            obj_parts.append(
                _wrap_underbrace(body, sym.descriptions[kind] if underbrace else None)
            )
            ref = _reference_latex(getattr(term, "reference", "measured"), sym, sup)
            defect_constraints.append(
                f"{var} = {sym.input}^{{{sup}}}_{{{k}}} - {ref}"
            )
        elif kind in ("INPUT_REG", "INPUT_RANDOM_WALK"):
            lo, hi = _proc_window(sym)
            sup = _idx_super(term.target_idx)
            dvar = _defect_sym(term)
            norm = f"{sym.norm_open}{dvar} {sym.norm_close}_{{2}}^{{2}}"
            body = _sum_prefix(lo, hi, sym) + f"{sym.reg_weight}\\, {norm}"
            if kind == "INPUT_RANDOM_WALK":
                desc_key = "INPUT_RANDOM_WALK"
                rhs = (
                    f"{sym.input}^{{{sup}}}_{{{k}}} - "
                    f"{sym.input}^{{{sup}}}_{{{k}-1}}"
                )
            else:
                trend = str(getattr(term, "trend", "FIRST_DIFF")).upper()
                if trend in ("FIRST_DIFF", "FIRST"):
                    desc_key = "FIRST_DIFF"
                    rhs = (
                        f"{sym.input}^{{{sup}}}_{{{k}}} - "
                        f"{sym.input}^{{{sup}}}_{{{k}-1}}"
                    )
                else:
                    desc_key = "SECOND_DIFF"
                    rhs = (
                        f"{sym.input}^{{{sup}}}_{{{k}}} - "
                        f"2 {sym.input}^{{{sup}}}_{{{k}-1}} + "
                        f"{sym.input}^{{{sup}}}_{{{k}-2}}"
                    )
            obj_parts.append(
                _wrap_underbrace(
                    body, sym.descriptions[desc_key] if underbrace else None
                )
            )
            defect_constraints.append(f"{dvar} = {rhs}")
        elif kind == "KNOWN_INPUT":
            pass
        elif kind == "UNKNOWN_INPUT":
            pass
        else:
            raise ValueError(f"Cannot render unknown term type: {kind!r}")

    constraints: list[str] = []
    dyn_rhs = (
        f"{sym.A} {sym.state}_{{{k}-1}} + {sym.B} {sym.input}_{{{k}-1}}"
    )
    if has_process:
        dyn_rhs += f" + {sym.process_noise}_{{{k}}}"
    constraints.append(f"{sym.state}_{{{k}}} = {dyn_rhs}")
    if has_meas:
        meas_rhs = (
            f"{sym.C} {sym.state}_{{{k}}} + {sym.D} {sym.input}_{{{k}}} "
            f"+ {sym.meas_noise}_{{{k}}}"
        )
        constraints.append(f"{sym.output}_{{{k}}} = {meas_rhs}")
    constraints.extend(defect_constraints)

    decision_vars = [sym.state, sym.input]
    if reg_terms:
        decision_vars.append(sym.defect)
    min_vars = ",\\, ".join(decision_vars)

    body_tex = _assemble_constrained(obj_parts, constraints, min_vars)
    result = _wrap_environment(body_tex, environment, multiline=True)
    if define_penalties:
        preamble = _penalty_definitions(terms, sym)
        if preamble:
            result = preamble + "\n" + result
    return result


def _assemble_constrained(
    obj_parts: list[str], constraints: list[str], min_vars: str
) -> str:
    """Build aligned ``minimize`` / ``subject to`` LaTeX rows."""
    obj = " + ".join(obj_parts) if obj_parts else "0"
    lines = [
        rf"& \underset{{{min_vars}}}{{\text{{minimize}}}} && {obj} \\",
    ]
    if constraints:
        lines.append(r"& \text{subject to} \\")
        last = len(constraints) - 1
        for i, c in enumerate(constraints):
            sep = "," if i < last else ""
            lines.append(rf"& && {c}{sep} \\")
        lines[-1] = lines[-1][: -len(r" \\")]
    else:
        lines[-1] = lines[-1][: -len(r" \\")]
    return "\n".join(lines)


def _min_subscript(sym: LatexSymbols, has_process: bool, has_input: bool) -> str:
    """Decision-variable list under ``\\min`` for substituted form."""
    lo, hi = _meas_window(sym)
    plo, phi = _proc_window(sym)
    parts = [f"{sym.state}_{{{lo}:{hi}}}"]
    if has_process:
        parts.append(f"{sym.process_noise}_{{{plo}:{phi}}}")
    if has_input:
        parts.append(f"{sym.input}_{{{plo}:{phi}}}")
    return ",\\, ".join(parts)


def _assemble(lines: list[str], min_vars: str, multiline: bool) -> str:
    """Join objective term lines under a ``\\min`` operator."""
    min_op = rf"\min_{{\substack{{{min_vars}}}}}"
    if not lines:
        return f"{min_op} \\; 0"
    if multiline:
        out = [f"{min_op} \\;"]
        out.append(f"&{lines[0]} \\\\")
        for line in lines[1:]:
            out.append(f"&+ {line} \\\\")
        # Drop trailing line break on the final row.
        out[-1] = out[-1][: -len(" \\\\")]
        return "\n".join(out)
    joined = lines[0]
    for line in lines[1:]:
        joined += f" + {line}"
    return f"{min_op} \\; {joined}"


def _wrap_environment(body: str, environment: str, multiline: bool) -> str:
    """Wrap body in ``equation`` / ``align`` or a bare ``aligned`` block."""
    if not environment:
        if multiline:
            return f"\\begin{{aligned}}\n{body}\n\\end{{aligned}}"
        return body
    inner = body
    if multiline:
        inner = f"\\begin{{aligned}}\n{body}\n\\end{{aligned}}"
    return f"\\begin{{{environment}}}\n{inner}\n\\end{{{environment}}}"


def _penalty_definitions(terms: list, sym: LatexSymbols) -> str:
    """Preamble defining Huber / dead-zone symbols used in the objective."""
    kinds = {_penalty_kind(t) for t in terms if hasattr(t, "penalty")}
    defs: list[str] = []
    if "HuberPenalty" in kinds:
        defs.append(
            r"\mathcal{H}_{\delta}(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ "
            r"\delta\left(|r| - \tfrac{1}{2}\delta\right) & |r| > \delta \end{cases}"
        )
    if "DeadzonePenalty" in kinds:
        defs.append(
            r"\mathcal{D}_{z}(r) = \tfrac{1}{2}\left(\max(0, |r| - z)\right)^2"
        )
    if not defs:
        return ""
    body = " \\\\\n".join(defs)
    return f"\\begin{{align}}\n{body}\n\\end{{align}}"


def _wrap_standalone(body: str) -> str:
    """Minimal ``article`` document wrapping ``body`` for direct compilation."""
    return (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}"
    )
