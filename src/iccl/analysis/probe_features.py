"""Named, indexed views of stored final-token representations."""

from dataclasses import dataclass
from typing import Any

import numpy as np

FAMILIES = ("residual_after_mixer", "residual_after_mlp", "mlp_hidden", "state", "head_read")


@dataclass(frozen=True)
class FeatureSelection:
    family: str
    name: str
    ranges: tuple[tuple[int, int], ...]

    @property
    def dimension(self) -> int:
        return sum(stop - start for start, stop in self.ranges)

    def select(self, array: np.ndarray) -> np.ndarray:
        flat = array.reshape(len(array), -1)
        return np.concatenate([flat[:, start:stop] for start, stop in self.ranges], axis=1)


def feature_selections(
    shapes: dict[str, list[int]],
    specification: dict[str, Any],
    head_groups: dict[str, list[list[int]]] | None = None,
) -> list[FeatureSelection]:
    """Flatten layer/head axes in C order without duplicating stored tensors."""
    result = []
    for family, modes in specification.items():
        if family not in FAMILIES or family not in shapes:
            raise ValueError(f"unknown or uncaptured representation {family}")
        shape = shapes[family]
        layers, layer_width = shape[0], int(np.prod(shape[1:]))
        for mode in modes:
            if mode == "all_layers":
                result.append(FeatureSelection(family, "all", ((0, layers * layer_width),)))
            elif mode == "each_layer":
                result.extend(
                    FeatureSelection(
                        family, f"L{layer + 1}", ((layer * layer_width, (layer + 1) * layer_width),)
                    )
                    for layer in range(layers)
                )
            elif mode == "each_head" and family in {"state", "head_read"}:
                heads, width = shape[1], int(np.prod(shape[2:]))
                result.extend(
                    FeatureSelection(
                        family,
                        f"L{layer + 1}H{head + 1}",
                        (((layer * heads + head) * width, (layer * heads + head + 1) * width),),
                    )
                    for layer in range(layers)
                    for head in range(heads)
                )
            else:
                raise ValueError(f"unsupported {family} selection: {mode}")
    for name, members in (head_groups or {}).items():
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "head group names must contain letters, digits, underscores or hyphens"
            )
        if not members or len({tuple(member) for member in members}) != len(members):
            raise ValueError(
                "head groups must be nonempty and contain distinct [layer, head] pairs"
            )
        for family in ("state", "head_read"):
            if family not in specification:
                continue
            layers, heads, *rest = shapes[family]
            width = int(np.prod(rest))
            ranges = []
            for member in members:
                if len(member) != 2 or any(type(x) is not int for x in member):
                    raise ValueError("head group members must be integer [layer, head] pairs")
                layer, head = member
                if not 1 <= layer <= layers or not 1 <= head <= heads:
                    raise ValueError("head group member is outside the captured architecture")
                start = ((layer - 1) * heads + head - 1) * width
                ranges.append((start, start + width))
            result.append(FeatureSelection(family, name, tuple(ranges)))
    if len({(item.family, item.name) for item in result}) != len(result):
        raise ValueError("duplicate feature selections")
    return result
