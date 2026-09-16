from .abc_score import parse
from .score_analysis import inspect_score


def _tick(value):
    return int(value * 256)


def viewer_data(score_abc, lyrics=""):
    info = inspect_score(score_abc, lyrics)
    score = parse(info["abc"])
    bars = [
        {
            "number": index + 1,
            "start": _tick(start),
            "duration": _tick(duration),
            "meter": f"{meter[0]}/{meter[1]}",
        }
        for index, (start, duration, meter) in enumerate(score.voices["Vocal"].bars)
    ]
    sections = []
    for section in info["roll"]["sections"]:
        first = section["start"]
        last = first + section["bars"]
        start_tick = bars[first]["start"]
        final_bar = bars[last - 1]
        end_tick = final_bar["start"] + final_bar["duration"]
        sections.append({
            **section,
            "start_bar": first + 1,
            "end_bar": last,
            "start_tick": start_tick,
            "end_tick": end_tick,
        })
    info["roll"]["bars"] = bars
    info["roll"]["sections"] = sections
    info["roll"]["total_ticks"] = bars[-1]["start"] + bars[-1]["duration"]
    return info


class HZ3_YuE2_ABCViewer:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "view"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = "Inspect the complete YuE2 ABC as a read-only piano roll and play Vocal, Ins, or both as a MIDI-style preview."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"score_abc": ("STRING", {"forceInput": True})},
            "optional": {"lyrics": ("STRING", {"forceInput": True})},
        }

    def view(self, score_abc, lyrics=""):
        data = viewer_data(score_abc, lyrics)
        return {"ui": {"abc_viewer": [data]}, "result": ()}


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_ABCViewer": HZ3_YuE2_ABCViewer}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_ABCViewer": "HZ3 YuE2 · ABC Piano Roll"}
