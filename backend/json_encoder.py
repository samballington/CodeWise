#!/usr/bin/env python3
"""
Custom JSON encoder that handles enums and other non-serializable types.
Prevents JSON serialization errors across the entire application.
"""

import json
from enum import Enum
from dataclasses import is_dataclass, asdict
from typing import Any


class EnumJSONEncoder(json.JSONEncoder):
    """JSON encoder that automatically converts enums to their values"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        elif is_dataclass(obj):
            # Convert dataclass to dict and recursively handle enums
            return self._convert_dict(asdict(obj))
        return super().default(obj)
    
    def _convert_dict(self, data: dict) -> dict:
        """Recursively convert enum values in nested dictionaries"""
        result = {}
        for key, value in data.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = self._convert_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    item.value if isinstance(item, Enum) else 
                    self._convert_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps with automatic enum handling"""
    return json.dumps(obj, cls=EnumJSONEncoder, **kwargs)


def to_json_dict(obj: Any) -> dict:
    """Convert any object to JSON-serializable dict with enum handling"""
    if is_dataclass(obj):
        encoder = EnumJSONEncoder()
        return encoder._convert_dict(asdict(obj))
    elif isinstance(obj, dict):
        encoder = EnumJSONEncoder()
        return encoder._convert_dict(obj)
    else:
        # For other types, convert via JSON round-trip to ensure serializability
        return json.loads(json_dumps(obj))