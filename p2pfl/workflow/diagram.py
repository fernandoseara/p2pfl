#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Workflow diagram generation (mermaid, ascii, matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from p2pfl.workflow.engine.stage import Stage
    from p2pfl.workflow.engine.workflow import Workflow


def _extract_transitions(
    stage_map: dict[str, Stage[Any]],
) -> tuple[dict[str, set[str | None]], set[str]]:
    """Extract transitions and dynamic stages from a stage map."""
    from p2pfl.workflow.validation import _extract_returns_from_run

    transitions: dict[str, set[str | None]] = {}
    dynamic_stages: set[str] = set()
    for name, stage in stage_map.items():
        extracted = _extract_returns_from_run(stage)
        transitions[name] = extracted.targets
        if extracted.dynamic_returns:
            dynamic_stages.add(name)
    return transitions, dynamic_stages


def generate_diagram(
    stage_map: dict[str, Stage[Any]],
    initial_stage: str,
    fmt: str = "mermaid",
) -> Any:
    """
    Generate a workflow diagram from the stage graph.

    Args:
        stage_map: Mapping of stage name to Stage instance.
        initial_stage: The entry-point stage name.
        fmt: Output format. ``"mermaid"`` (default), ``"ascii"``, or ``"image"``.

    Returns:
        For ``"mermaid"`` and ``"ascii"``: the diagram as a string.
        For ``"image"``: a ``matplotlib.figure.Figure`` object.

    """
    transitions, dynamic_stages = _extract_transitions(stage_map)

    if fmt == "mermaid":
        return _mermaid_diagram(transitions, initial_stage, dynamic_stages)
    elif fmt == "ascii":
        return _ascii_diagram(transitions, initial_stage, dynamic_stages)
    elif fmt == "image":
        return _image_diagram(transitions, initial_stage, dynamic_stages)
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'mermaid', 'ascii', or 'image'.")


def _mermaid_diagram(
    transitions: dict[str, set[str | None]],
    initial_stage: str,
    dynamic_stages: set[str],
) -> str:
    """Generate a Mermaid flowchart diagram."""
    lines = ["graph LR"]

    # Mark initial stage
    lines.append(f'    {_mermaid_id(initial_stage)}(["**{initial_stage}**"])')

    # Declare nodes
    for name in transitions:
        if name != initial_stage:
            lines.append(f'    {_mermaid_id(name)}["{name}"]')

    # End node
    lines.append(f"    {_mermaid_id('__end__')}((\"END\"))")

    # Edges
    for name, targets in transitions.items():
        for target in sorted(targets, key=lambda x: x or ""):
            if target is None:
                lines.append(f"    {_mermaid_id(name)} --> {_mermaid_id('__end__')}")
            else:
                lines.append(f"    {_mermaid_id(name)} --> {_mermaid_id(target)}")

    # Dynamic return annotations
    for name in dynamic_stages:
        lines.append(f"    {_mermaid_id(name)} -.->|dynamic| {_mermaid_id('__unknown__')}[\"?\"]")

    return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Sanitize a stage name for use as a Mermaid node ID."""
    return name.replace(" ", "_").replace("-", "_")


def _ascii_diagram(
    transitions: dict[str, set[str | None]],
    initial_stage: str,
    dynamic_stages: set[str],
) -> str:
    """Generate a simple ASCII diagram."""
    lines: list[str] = []
    lines.append(f"Workflow: [{initial_stage}] (start)")
    lines.append("")

    # BFS order for readability
    visited: set[str] = set()
    queue = [initial_stage]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        targets = transitions.get(current, set())
        target_strs: list[str] = []
        for t in sorted(targets, key=lambda x: x or "~~~"):
            if t is None:
                target_strs.append("[END]")
            else:
                target_strs.append(t)
                if t not in visited:
                    queue.append(t)

        arrow = " --> " + ", ".join(target_strs) if target_strs else " --> (no returns found)"
        prefix = "* " if current in dynamic_stages else "  "
        lines.append(f"{prefix}{current}{arrow}")

    # Legend
    unreachable = set(transitions.keys()) - visited
    if unreachable:
        lines.append("")
        lines.append(f"  UNREACHABLE: {', '.join(sorted(unreachable))}")

    if dynamic_stages:
        lines.append("")
        lines.append("  * = has dynamic returns (cannot fully validate)")

    return "\n".join(lines)


def _image_diagram(
    transitions: dict[str, set[str | None]],
    initial_stage: str,
    dynamic_stages: set[str],
) -> Any:
    """Generate a polished matplotlib Figure of the workflow graph."""
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import networkx as nx  # type: ignore[import-untyped]

    G = nx.DiGraph()
    for name in transitions:
        G.add_node(name)
    G.add_node("END")
    for name, targets in transitions.items():
        for target in targets:
            G.add_edge(name, "END" if target is None else target)

    # -- Classify edges as forward or back --
    layers: dict[str, int] = {initial_stage: 0}
    queue = [initial_stage]
    while queue:
        cur = queue.pop(0)
        for succ in G.successors(cur):
            if succ not in layers:
                layers[succ] = layers[cur] + 1
                queue.append(succ)
    mx = max(layers.values(), default=0)
    for n in G.nodes():
        if n not in layers:
            mx += 1
            layers[n] = mx

    back_edges: set[tuple[str, str]] = set()
    for u, v in G.edges():
        if layers.get(v, 0) <= layers.get(u, 0):
            back_edges.add((u, v))

    # -- Layered layout --
    layer_groups: dict[int, list[str]] = {}
    for n, lay in layers.items():
        layer_groups.setdefault(lay, []).append(n)

    x_gap = 2.4
    y_gap = 1.4
    pos: dict[str, tuple[float, float]] = {}
    for lay, nodes in layer_groups.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (lay * x_gap, -(i - (len(nodes) - 1) / 2) * y_gap)

    # -- Palette (muted, modern) --
    palette = {
        "node_fill": "#5b7fa5",
        "node_border": "#4a6d8c",
        "initial_fill": "#4a7c6f",
        "initial_border": "#3d6b5e",
        "end_fill": "#8c9bab",
        "end_border": "#7a8999",
        "dynamic_fill": "#c4805a",
        "dynamic_border": "#a96d4d",
        "edge": "#7a8899",
        "back_edge": "#a0aab5",
        "text": "#ffffff",
        "shadow": "#00000018",
    }

    # -- Figure sizing --
    n_layers = max(layers.values(), default=0) + 1
    max_height = max(len(v) for v in layer_groups.values())
    fig_w = max(7.0, n_layers * 2.4 + 1.5)
    fig_h = max(2.5, max_height * 1.4 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # -- Node dimensions --
    node_w, node_h = 1.5, 0.55
    end_radius = 0.22

    # -- Draw nodes --
    for node, (x, y) in pos.items():
        if node == "END":
            # Small subtle terminal dot
            shadow: mpatches.Patch = mpatches.Circle(
                (x + 0.02, y - 0.02),
                end_radius,
                facecolor=palette["shadow"],
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(shadow)
            dot = mpatches.Circle(
                (x, y),
                end_radius,
                facecolor=palette["end_fill"],
                edgecolor=palette["end_border"],
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(dot)
            ax.text(
                x,
                y,
                "END",
                ha="center",
                va="center",
                fontsize=7,
                fontfamily="sans-serif",
                fontweight="bold",
                color=palette["text"],
                zorder=4,
            )
            continue

        # Pick fill / border for this node
        if node == initial_stage:
            fill = palette["initial_fill"]
            border = palette["initial_border"]
            border_lw = 2.2
        elif node in dynamic_stages:
            fill = palette["dynamic_fill"]
            border = palette["dynamic_border"]
            border_lw = 1.4
        else:
            fill = palette["node_fill"]
            border = palette["node_border"]
            border_lw = 1.4

        # Subtle drop shadow
        shadow = mpatches.FancyBboxPatch(
            (x - node_w / 2 + 0.03, y - node_h / 2 - 0.03),
            node_w,
            node_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.12),
            facecolor=palette["shadow"],
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(shadow)

        # Node rectangle
        rect = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w,
            node_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.12),
            facecolor=fill,
            edgecolor=border,
            linewidth=border_lw,
            zorder=3,
        )
        ax.add_patch(rect)

        # Label
        label = node.replace("_", " ")
        txt = ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontfamily="sans-serif",
            fontweight="bold",
            color=palette["text"],
            zorder=4,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="#00000030")])

    # -- Helper to compute connection points on node borders --
    def _edge_point(node: str, toward: tuple[float, float], outgoing: bool) -> tuple[float, float]:
        """Return the point on the node border facing *toward*."""
        nx_, ny = pos[node]
        tx, ty = toward
        if node == "END":
            # Circle edge point
            import math

            dx, dy = tx - nx_, ty - ny
            dist = math.hypot(dx, dy) or 1.0
            return (nx_ + dx / dist * end_radius, ny + dy / dist * end_radius)
        # Rounded rectangle: exit from left/right edge
        hw = node_w / 2 + 0.12  # account for pad
        if outgoing:
            return (nx_ + hw, ny) if tx >= nx_ else (nx_ - hw, ny)
        return (nx_ - hw, ny) if tx <= nx_ else (nx_ + hw, ny)

    # -- Draw edges --
    # Sort back-edges by horizontal span so we can offset them
    import matplotlib.path as mpath

    back_edge_list = sorted(back_edges, key=lambda e: abs(pos[e[0]][0] - pos[e[1]][0]))
    all_ys = [p[1] for p in pos.values()]
    bottom_y = min(all_ys)
    hh = node_h / 2 + 0.12

    for u, v in G.edges():
        is_back = (u, v) in back_edges
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Connection points
        pt_start = _edge_point(u, (x1, y1), outgoing=True)
        pt_end = _edge_point(v, (x0, y0), outgoing=False)

        if is_back:
            # Offset index so multiple back-edges don't overlap
            idx = back_edge_list.index((u, v))
            offset = idx * 0.3

            # Tight curve just below the lowest nodes
            curve_depth = bottom_y - 0.45 - offset

            # Exit from bottom of source, enter bottom of target
            # Offset entry x slightly per index to separate arrowheads
            pt_s = (x0, y0 - hh)
            pt_e = (x1 + 0.25 * idx, y1 - hh)

            verts = [
                pt_s,
                (x0, curve_depth),
                (x1, curve_depth),
                pt_e,
            ]
            codes = [1, 4, 4, 4]  # MOVETO, CURVE4 x3

            path = mpath.Path(verts, codes)
            patch = mpatches.FancyArrowPatch(
                path=path,
                arrowstyle="-|>,head_length=6,head_width=3",
                color=palette["back_edge"],
                linewidth=1.3,
                linestyle=(0, (5, 3)),
                zorder=1,
            )
            ax.add_patch(patch)
        else:
            # Forward edge
            dx = x1 - x0
            dy = y1 - y0
            # Slight curve for edges that aren't purely horizontal
            rad = 0.0
            if abs(dy) > 0.3 and abs(dx) > 0.1:
                rad = 0.15 if dy < 0 else -0.15

            ax.annotate(
                "",
                xy=pt_end,
                xytext=pt_start,
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
                    "color": palette["edge"],
                    "lw": 1.5,
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )

    # -- Margins --
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    # Account for back-edge curves that go below
    back_depth = 0.45 + len(back_edges) * 0.3 if back_edges else 0.0
    min_y = min(ys) - back_depth - 0.3
    pad_x, pad_y = 1.2, 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min_y - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.3)
    return fig


def diagram(workflow: Workflow[Any], fmt: str = "mermaid") -> Any:
    """
    Generate a workflow diagram.

    Convenience wrapper around ``generate_diagram`` that accepts a
    ``Workflow`` instance directly.

    Args:
        workflow: The workflow to diagram.
        fmt: ``"mermaid"`` (default), ``"ascii"``, or ``"image"``.

    Returns:
        For ``"mermaid"``/``"ascii"``: the diagram as a string.
        For ``"image"``: a ``matplotlib.figure.Figure``.

    """
    stage_map = {s.name: s for s in workflow.get_stages()}
    return generate_diagram(stage_map, workflow.initial_stage, fmt=fmt)
