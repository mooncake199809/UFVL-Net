# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, ARCHITECTURE, HEADS,
                      build_backbone, build_architecture, build_head)
from .architecture import *  # noqa: F401,F403
from .heads import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'ARCHITECTURE', 'build_backbone',
    'build_head', 'build_architecture'
]
