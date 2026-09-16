"""Generic list <-> batch helpers ported from gitmylo/FlowNodes.

ComfyUI's classic node API supports two flags that let a node consume or expand
Python-list values across sockets:

- INPUT_IS_LIST: the node receives the connected sources as a Python list.
- OUTPUT_IS_LIST: the value the node returns is expanded into one output socket
  per list item.

These nodes are type-agnostic (the "*" any-type trick), so they work for AUDIO,
STRING, CONDITIONING, etc., and are useful to bundle several values into a batch
and then fan them out again for per-item processing.
"""

from __future__ import annotations


class AnyType(str):
    """String type whose __ne__ always returns False so *any* socket type
    connects. Credits: pythongosssss/ComfyUI-Custom-Scripts repeater hack."""

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class HZ3_YuE2_ListToBatch:
    CATEGORY = "HZ3 YuE2/List"
    FUNCTION = "merge"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("Batch",)
    INPUT_IS_LIST = (True,)
    DESCRIPTION = "List to Batch: consume the incoming values as a list and emit them as a single 'batch' value. Type-agnostic (works for AUDIO, STRING, CONDITIONING, ...)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"List": (any,)}}

    def merge(self, List, **kwargs):
        return (List,)


class HZ3_YuE2_BatchToList:
    CATEGORY = "HZ3 YuE2/List"
    FUNCTION = "unmerge"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("List",)
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Batch to List: take a list-valued 'batch' and expand it into one output socket per item (List[0], List[1], ...). Type-agnostic."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"Batch": (any,)}}

    def unmerge(self, Batch, **kwargs):
        return (Batch,)


class HZ3_YuE2_CreateBatch:
    CATEGORY = "HZ3 YuE2/List"
    FUNCTION = "create"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("Batch",)
    DESCRIPTION = "Create a batch: collect up to 20 input values (in order) into a single list 'batch' value."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"no_cache": ("NO_CACHE",)},
            "optional": {f"input{index}": (any, {"dynamicAvailable": "input"}) for index in range(20)},
        }

    def create(self, **kwargs):
        return (list(value for key, value in kwargs.items() if key != "no_cache"),)


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ListToBatch": HZ3_YuE2_ListToBatch,
    "HZ3_YuE2_BatchToList": HZ3_YuE2_BatchToList,
    "HZ3_YuE2_CreateBatch": HZ3_YuE2_CreateBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ListToBatch": "HZ3 YuE2 · List to Batch",
    "HZ3_YuE2_BatchToList": "HZ3 YuE2 · Batch to List",
    "HZ3_YuE2_CreateBatch": "HZ3 YuE2 · Create Batch",
}
