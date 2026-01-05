"""
Scorer package for navigation and grid representation.

This package provides:
- Grid representation for OSM data
- Graph context and navigation
- Presentation formats (textual, JSON, graphviz)
- POI extraction and matching
- Agent orchestration
"""

# Presentation formats (no external dependencies beyond dataclasses/json)
from scorer.presentation_formats import (
    generate_textual_representation,
    generate_json_representation,
    generate_json_representation_dict,
    generate_graphviz_representation,
    generate_all_representations,
    NavigationStepGenerator,
    RepresentationFormat,
    generate_batch_request,
    extract_presentation_context,
    NodeConnection,
    NodeInfo,
    PresentationContext
)

# Try to import grid_repsentation (requires numpy)
try:
    from scorer.grid_repsentation import (
        convert2grid,
        setup_grid_dynamically,
        find_poi_grid_positions,
        get_node_id_from_position,
        calculate_distance,
        calculate_heading_from_coords,
        get_delta_from_heading,
        get_relative_direction_from_heading
    )
except ImportError:
    convert2grid = None
    setup_grid_dynamically = None
    find_poi_grid_positions = None
    get_node_id_from_position = None
    calculate_distance = None
    calculate_heading_from_coords = None
    get_delta_from_heading = None
    get_relative_direction_from_heading = None

# Try to import graph_context (requires toon_format)
try:
    from scorer.graph_context import (
        navigate,
        build_navigation_context,
        encode_navigation_context,
        decode_navigation_context,
        encode_for_decision
    )
except ImportError:
    navigate = None
    build_navigation_context = None
    encode_navigation_context = None
    decode_navigation_context = None
    encode_for_decision = None

__all__ = [
    # Presentation formats (always available)
    'generate_textual_representation',
    'generate_json_representation',
    'generate_json_representation_dict',
    'generate_graphviz_representation',
    'generate_all_representations',
    'NavigationStepGenerator',
    'RepresentationFormat',
    'generate_batch_request',
    'extract_presentation_context',
    'NodeConnection',
    'NodeInfo',
    'PresentationContext',
    # Grid representation (requires numpy)
    'convert2grid',
    'setup_grid_dynamically',
    'find_poi_grid_positions',
    'get_node_id_from_position',
    'calculate_distance',
    'calculate_heading_from_coords',
    'get_delta_from_heading',
    'get_relative_direction_from_heading',
    # Graph context (requires toon_format)
    'navigate',
    'build_navigation_context',
    'encode_navigation_context',
    'decode_navigation_context',
    'encode_for_decision',
]
