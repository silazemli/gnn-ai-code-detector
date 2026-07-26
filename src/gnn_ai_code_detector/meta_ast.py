from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field

class NodeKind(Enum):
    IF = auto()
    FOR = auto()

@dataclass
class MetaNode:
    kind: NodeKind
    original_kind: str
    children: list[MetaNode] = field(default_factory=list)

@dataclass
class MetaAST:
    root: MetaNode