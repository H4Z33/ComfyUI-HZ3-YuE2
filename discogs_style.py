import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

import folder_paths

from .score_analysis import inspect_score


def _model_dirs():
    """Resolve every ComfyUI model root, including extra_model_paths entries."""
    directories = []
    try:
        directories.extend(Path(path) / "discogs_effnet" for path in folder_paths.get_folder_paths("audio_classifiers"))
    except KeyError:
        pass
    directories.extend([
        Path(folder_paths.models_dir) / "audio_classifiers" / "discogs_effnet",
        Path.home() / "AppData/Local/Comfy-Desktop/ComfyUI-Shared/models/audio_classifiers/discogs_effnet",
    ])
    return list(dict.fromkeys(directories))


def _model_files():
    directories = _model_dirs()
    for directory in directories:
        model = directory / "discogs-effnet-bsdynamic-1.onnx"
        metadata = directory / "discogs-effnet-bsdynamic-1.json"
        if model.exists() and metadata.exists():
            return model, metadata
    raise FileNotFoundError("Discogs-EffNet files were not found in: " + ", ".join(map(str, directories)))


def _patches(audio, sample_rate):
    audio = audio.detach().float().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(0)
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=512, hop_length=256,
        f_min=0, f_max=8000, n_mels=96, power=2.0,
        center=True, norm="slaney", mel_scale="slaney",
    )(audio)
    mel = torch.log10(1.0 + 10000.0 * mel).transpose(0, 1)
    if mel.shape[0] < 128:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 128 - mel.shape[0]))
    starts = list(range(0, max(1, mel.shape[0] - 127), 64))
    if starts[-1] + 128 < mel.shape[0]:
        starts.append(mel.shape[0] - 128)
    # Uniformly cap very long songs without biasing the beginning.
    if len(starts) > 192:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, 192).round().astype(int)]
    return torch.stack([mel[start:start + 128] for start in starts]).numpy().astype(np.float32)


def _score_suffix(score_abc):
    if not score_abc or not score_abc.strip():
        return ""
    info = inspect_score(score_abc)
    return f", {info['bpm']} BPM, {info['meter']}, key of {info['key']}"


def _style_label(label):
    return label.split("---", 1)[-1]


def _segment_slices(item_count, segment_count):
    """Return non-empty, contiguous slices covering every model window once."""
    segment_count = max(1, min(int(segment_count), int(item_count)))
    edges = np.linspace(0, item_count, segment_count + 1).round().astype(int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(segment_count)
            if edges[i] < edges[i + 1]]


class HZ3_YuE2_AudioToStyle:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "classify"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("style", "analysis")
    OUTPUT_NODE = True
    DESCRIPTION = "Classify the actual audio with the official Discogs-EffNet 400-style model. The exact visible prompt is also emitted as style for YuE2. No manual genre or intent fields."

    _session = None
    _classes = None
    _heads = {}

    HEADS = {
        "voice": "voice_instrumental-discogs-effnet-1",
        "gender": "gender-discogs-effnet-1",
        "instruments": "mtg_jamendo_instrument-discogs-effnet-1",
        "moods": "mtg_jamendo_moodtheme-discogs-effnet-1",
        "danceability": "danceability-discogs-effnet-1",
        "tonality": "tonal_atonal-discogs-effnet-1",
        "timbre": "timbre-discogs-effnet-1",
        "acoustic": "mood_acoustic-discogs-effnet-1",
        "electronic": "mood_electronic-discogs-effnet-1",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "top_styles": ("INT", {"default": 6, "min": 1, "max": 20}),
                "top_instruments": ("INT", {"default": 6, "min": 0, "max": 20}),
                "top_moods": ("INT", {"default": 6, "min": 0, "max": 20}),
                "minimum_score": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum probability for multi-label genre, instrument and mood results."}),
                "analysis_mode": (["global", "segmented"], {"default": "global", "tooltip": "Global preserves the original whole-song analysis. Segmented adds a donor timeline to the analysis output consumed by MixMash."}),
                "timeline_segments": ("INT", {"default": 6, "min": 2, "max": 12, "tooltip": "Number of chronological regions reported in segmented mode."}),
            },
            "optional": {
                "score_abc": ("STRING", {"forceInput": True, "tooltip": "Optional: append BPM, meter and key measured by SheetSage."}),
            },
        }

    @classmethod
    def _load(cls):
        model_path, metadata_path = _model_files()
        if cls._session is None:
            import onnxruntime as ort
            cls._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            cls._classes = json.loads(metadata_path.read_text(encoding="utf-8"))["classes"]
        return cls._session, cls._classes

    @classmethod
    def _head(cls, name):
        if name not in cls._heads:
            import onnxruntime as ort
            stem = cls.HEADS[name]
            for directory in _model_dirs():
                model, metadata = directory / f"{stem}.onnx", directory / f"{stem}.json"
                if model.exists() and metadata.exists():
                    cls._heads[name] = (
                        ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]),
                        json.loads(metadata.read_text(encoding="utf-8"))["classes"],
                    )
                    break
            else:
                raise FileNotFoundError(f"Discogs-EffNet classification head is missing: {stem}")
        return cls._heads[name]

    @classmethod
    def _predict_head_raw(cls, name, embeddings):
        session, labels = cls._head(name)
        values = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: embeddings})[0]
        return labels, values

    @classmethod
    def _predict_head(cls, name, embeddings):
        labels, values = cls._predict_head_raw(name, embeddings)
        return labels, values.mean(axis=0)

    @staticmethod
    def _rank(labels, values, count, threshold):
        return [(labels[i], float(values[i])) for i in np.argsort(values)[::-1] if values[i] >= threshold][:count]

    @staticmethod
    def _binary_summary(head_results):
        summary = {}
        for name in ("voice", "gender", "danceability", "tonality", "timbre", "acoustic", "electronic"):
            labels, values = head_results[name]
            index = int(np.argmax(values))
            summary[name] = (labels[index], float(values[index]), {label: float(value) for label, value in zip(labels, values)})
        return summary

    def classify(self, audio, analysis_mode="global", timeline_segments=6, top_styles=6,
                 top_instruments=6, top_moods=6, minimum_score=0.08, score_abc=""):
        session, classes = self._load()
        patches = _patches(audio["waveform"], int(audio["sample_rate"]))
        predictions, embeddings = session.run(["activations", "embeddings"], {"melspectrogram": patches})
        scores = predictions.mean(axis=0)
        ranked = np.argsort(scores)[::-1]
        selected = [i for i in ranked if scores[i] >= minimum_score][:top_styles]
        if not selected:
            selected = list(ranked[:top_styles])
        genre_tags = list(dict.fromkeys(_style_label(classes[i]) for i in selected))

        raw_heads = {name: self._predict_head_raw(name, embeddings) for name in self.HEADS}
        head_results = {name: (labels, values.mean(axis=0)) for name, (labels, values) in raw_heads.items()}
        instrument_tags = self._rank(*head_results["instruments"], top_instruments, minimum_score)
        mood_tags = self._rank(*head_results["moods"], top_moods, minimum_score)
        binary = self._binary_summary(head_results)

        descriptors = []
        if binary["voice"][0] == "voice":
            descriptors.append(f"{binary['gender'][0]} vocals")
        else:
            descriptors.append("instrumental")
        descriptors.extend(label for label, _ in instrument_tags if label != "voice")
        descriptors.extend(label for label, _ in mood_tags)
        if binary["danceability"][0] == "danceable": descriptors.append("danceable")
        descriptors.append(binary["tonality"][0])
        descriptors.append(f"{binary['timbre'][0]} timbre")
        if binary["acoustic"][2].get("acoustic", 0) >= .55: descriptors.append("acoustic production")
        if binary["electronic"][2].get("electronic", 0) >= .55: descriptors.append("electronic production")
        style = ", ".join(dict.fromkeys(genre_tags + descriptors)) + _score_suffix(score_abc)

        sections = ["STYLES"] + [f"{classes[i].replace('---', ' / ')}: {scores[i]:.1%}" for i in selected]
        sections += ["", "INSTRUMENTS"] + [f"{label}: {score:.1%}" for label, score in instrument_tags]
        sections += ["", "MOOD / THEME"] + [f"{label}: {score:.1%}" for label, score in mood_tags]
        sections += ["", "BINARY ATTRIBUTES"] + [f"{name}: {label} ({score:.1%})" for name, (label, score, _) in binary.items()]
        if analysis_mode == "segmented":
            waveform = audio["waveform"]
            duration = float(waveform.shape[-1]) / float(audio["sample_rate"])
            slices = _segment_slices(len(embeddings), timeline_segments)
            sections += ["", "DONOR TIMELINE (chronological model-window analysis)"]
            for number, window in enumerate(slices, 1):
                start_fraction = window.start / len(embeddings)
                end_fraction = window.stop / len(embeddings)
                segment_scores = predictions[window].mean(axis=0)
                segment_genres = self._rank(classes, segment_scores, min(3, top_styles), minimum_score)
                segment_heads = {
                    name: (labels, values[window].mean(axis=0))
                    for name, (labels, values) in raw_heads.items()
                }
                segment_instruments = self._rank(*segment_heads["instruments"], min(4, top_instruments), minimum_score)
                segment_moods = self._rank(*segment_heads["moods"], min(4, top_moods), minimum_score)
                segment_binary = self._binary_summary(segment_heads)
                genre_text = ", ".join(f"{_style_label(label)} {score:.0%}" for label, score in segment_genres) or "no confident genre tag"
                instrument_text = ", ".join(f"{label} {score:.0%}" for label, score in segment_instruments) or "no confident instrument tag"
                mood_text = ", ".join(f"{label} {score:.0%}" for label, score in segment_moods) or "no confident mood tag"
                attributes = ", ".join(
                    f"{name}={label} ({score:.0%})"
                    for name, (label, score, _) in segment_binary.items()
                )
                sections += [
                    f"Segment {number}: {start_fraction:.0%}-{end_fraction:.0%} / {duration * start_fraction:.1f}-{duration * end_fraction:.1f}s",
                    f"  styles: {genre_text}",
                    f"  instruments: {instrument_text}",
                    f"  mood/theme: {mood_text}",
                    f"  attributes: {attributes}",
                ]
        details = "Discogs-EffNet analysis\n\n" + "\n".join(sections) + f"\n\nYuE2 style sent:\n{style}"
        return {"ui": {"text": [details]}, "result": (style, details)}
