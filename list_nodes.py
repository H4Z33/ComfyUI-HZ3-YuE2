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


class HZ3_YuE2_BatchGetItem:
    CATEGORY = "HZ3 YuE2/List"
    FUNCTION = "get_item"
    RETURN_TYPES = (any, "INT")
    RETURN_NAMES = ("item", "count")
    DESCRIPTION = (
        "Get one item from a batch by index (negative = from the end). Works on a "
        "batched AUDIO (returns a single clip as an AUDIO with batch=1), on a Python "
        "list/tuple, and on other batched tensors. The item comes out on output [0]."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": (any,),
                "index": ("INT", {"default": 0, "min": -99999, "max": 99999,
                                  "tooltip": "0-based index; negative counts from the end (e.g. -1 = last)."}),
            },
        }

    def get_item(self, batch, index, **kwargs):
        # Batched AUDIO: a dict with waveform + sample_rate.
        if isinstance(batch, dict) and "waveform" in batch and "sample_rate" in batch:
            waveform = batch["waveform"]
            count = waveform.shape[0]
            i = index + count if index < 0 else index
            i = max(0, min(count - 1, i))
            clip = waveform[i:i + 1]                     # keep batch dim = 1
            return ({"waveform": clip, "sample_rate": batch["sample_rate"]}, count)

        # Python list/tuple.
        if isinstance(batch, (list, tuple)):
            count = len(batch)
            i = index + count if index < 0 else index
            i = max(0, min(count - 1, i))
            return (batch[i], count)

        # Other tensor-like batch (IMAGE, LATENT, ...).
        if hasattr(batch, "shape") and batch.ndim >= 1:
            count = batch.shape[0]
            i = index + count if index < 0 else index
            i = max(0, min(count - 1, i))
            return (batch[i], count)

        raise ValueError("batch must be a batched AUDIO, a list/tuple, or a 1D+ tensor.")


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ListToBatch": HZ3_YuE2_ListToBatch,
    "HZ3_YuE2_BatchToList": HZ3_YuE2_BatchToList,
    "HZ3_YuE2_CreateBatch": HZ3_YuE2_CreateBatch,
    "HZ3_YuE2_BatchGetItem": HZ3_YuE2_BatchGetItem,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ListToBatch": "HZ3 YuE2 · List to Batch",
    "HZ3_YuE2_BatchToList": "HZ3 YuE2 · Batch to List",
    "HZ3_YuE2_CreateBatch": "HZ3 YuE2 · Create Batch",
    "HZ3_YuE2_BatchGetItem": "HZ3 YuE2 · Get Batch Item",
}
