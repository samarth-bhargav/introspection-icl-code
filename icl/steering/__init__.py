"""Steering vector subsystem: concept libraries, injection hooks, and range calibration."""

from .concepts import (  # noqa: F401
    ConceptLibrary,
    ConceptSpec,
    DEFAULT_CONCEPTS,
    format_as_chat,
)
from .injection import (  # noqa: F401
    find_trigger_positions,
    forward_with_multi_layer_steering,
    forward_with_positional_steering,
    generate_with_steering,
)
from .ranges import (  # noqa: F401
    SteeringRanges,
    compute_max_strength,
    compute_max_strength_multilayer,
)
