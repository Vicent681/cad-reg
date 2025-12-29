#!/usr/bin/env python3
"""Dump DXF entities to stdout.

This script relies on ``ezdxf`` to read a DXF file and prints a
human-readable summary of every entity found in all layouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VectorTypes = tuple[type[object], ...]
DEFAULT_DXF_PATH = Path("/Users/vincent/Work/危大图纸/3、6#楼立面图_t3.dxf")

SPECIAL_ENTITY_ATTRS: dict[str, tuple[str, ...]] = {
    "LINE": ("start", "end"),
    "CIRCLE": ("center", "radius"),
    "ARC": ("center", "radius", "start_angle", "end_angle"),
    "POINT": ("location",),
    "LWPOLYLINE": ("points",),
    "POLYLINE": ("points",),
    "TEXT": ("insert", "text", "height"),
    "MTEXT": ("insert", "text", "char_height"),
    "ELLIPSE": ("center", "major_axis", "ratio", "start_param", "end_param"),
    "SPLINE": ("fit_points", "control_points"),
}


def ensure_ezdxf() -> tuple[Any, VectorTypes]:
    """Import ezdxf lazily so ``-h`` works even if it's missing."""

    try:
        import ezdxf  # type: ignore
        from ezdxf.math import Vec2, Vec3  # type: ignore
    except ImportError as exc:  # pragma: no cover - triggered at runtime
        raise SystemExit(
            "ezdxf is required. Install it with 'pip install ezdxf'."
        ) from exc

    return ezdxf, (Vec2, Vec3)


def normalize_value(value: Any, vector_types: VectorTypes) -> Any:
    """Convert ezdxf specific objects into JSON serializable values."""

    if isinstance(value, vector_types):
        return tuple(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v, vector_types) for v in value]
    if isinstance(value, dict):
        return {
            key: normalize_value(val, vector_types) for key, val in value.items()
        }
    return value


def describe_entity(
    entity: Any, vector_types: VectorTypes, ezdxf_mod: Any
) -> dict[str, Any]:
    """Return a serializable description of an entity."""

    data = {
        "type": entity.dxftype(),
        "handle": entity.dxf.handle,
    }

    try:
        raw_attrs = entity.dxfattribs()
    except ezdxf_mod.DXFStructureError:
        raw_attrs = {}

    data["attributes"] = {
        key: normalize_value(value, vector_types)
        for key, value in raw_attrs.items()
    }
    return data


def describe_layer(
    layer: Any, vector_types: VectorTypes, ezdxf_mod: Any
) -> dict[str, Any]:
    """Return a serializable description of a layer definition."""

    data = {
        "record": "layer",
        "name": layer.dxf.name,
        "handle": getattr(layer.dxf, "handle", None),
    }

    try:
        raw_attrs = layer.dxfattribs()
    except ezdxf_mod.DXFStructureError:
        raw_attrs = {}

    data["attributes"] = {
        key: normalize_value(value, vector_types)
        for key, value in raw_attrs.items()
    }
    return data


def format_value_for_print(value: Any) -> str:
    """Return a readable string for nested attribute values."""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_layer_summary(layer_desc: dict[str, Any]) -> str:
    """Return a single-line human readable layer summary."""

    header = f"[LAYER] {layer_desc.get('name')} (handle={layer_desc.get('handle')})"
    attributes = layer_desc.get("attributes", {})
    extras = []
    for key in ("color", "linetype", "lineweight", "plot"):
        value = attributes.get(key)
        if value not in (None, ""):
            extras.append(f"{key}={format_value_for_print(value)}")
    if not extras:
        return header
    return f"{header}: " + ", ".join(extras)


def format_entity_summary(entity_desc: dict[str, Any]) -> str:
    """Return a single-line human readable entity summary."""

    header = (
        f"[{entity_desc.get('layout', '?')}] {entity_desc.get('type')} "
        f"(handle={entity_desc.get('handle')})"
    )
    attributes = entity_desc.get("attributes", {})

    parts = []
    layer_name = attributes.get("layer")
    if layer_name:
        parts.append(f"layer={layer_name}")
    color = attributes.get("color") or attributes.get("true_color") or attributes.get(
        "aci_color"
    )
    if color is not None:
        parts.append(f"color={color}")
    linetype = attributes.get("linetype")
    if linetype:
        parts.append(f"linetype={linetype}")
    lineweight = attributes.get("lineweight")
    if lineweight:
        parts.append(f"lineweight={lineweight}")

    special_attr_names = SPECIAL_ENTITY_ATTRS.get(entity_desc.get("type", ""), ())
    for attr_name in special_attr_names:
        if attr_name not in attributes:
            continue
        parts.append(f"{attr_name}={format_value_for_print(attributes[attr_name])}")

    if not parts:
        return header
    return f"{header}: " + ", ".join(parts)


def iter_layout_entities(doc: Any):
    """Yield tuples of (layout name, entity)."""

    for layout in doc.layouts:
        # Skip virtual modelspace copies to avoid duplication.
        if layout.name.startswith("*Model_"):
            continue
        for entity in layout:
            yield layout.name, entity


def load_document(path: Path, ezdxf_mod: Any) -> Any:
    try:
        return ezdxf_mod.readfile(path)
    except (IOError, ezdxf_mod.DXFStructureError) as exc:
        raise SystemExit(f"Failed to read '{path}': {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a DXF file and print each entity in a human-readable format."
    )
    parser.add_argument(
        "--dxf-path",
        dest="dxf_path",
        type=Path,
        default=DEFAULT_DXF_PATH,
        help=f"Path to the DXF file to examine (default: {DEFAULT_DXF_PATH})",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print a compact JSON array instead of line-by-line output.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Include layer definitions in the output before entity data.",
    )
    args = parser.parse_args()

    ezdxf_mod, vector_types = ensure_ezdxf()
    doc = load_document(args.dxf_path, ezdxf_mod)

    layer_descriptions = []
    if args.list_layers:
        layer_descriptions = [
            describe_layer(layer, vector_types, ezdxf_mod)
            for layer in doc.layers
        ]

    entities = []
    for layout_name, entity in iter_layout_entities(doc):
        description = describe_entity(entity, vector_types, ezdxf_mod)
        description["layout"] = layout_name
        entities.append(description)

    if args.compact:
        if args.list_layers:
            payload = {
                "layers": layer_descriptions,
                "entities": entities,
            }
        else:
            payload = entities
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if args.list_layers:
            for layer in layer_descriptions:
                print(format_layer_summary(layer))
        for entry in entities:
            print(format_entity_summary(entry))


if __name__ == "__main__":
    main()
