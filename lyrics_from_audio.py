"""Find catalog lyrics for an audio input via AcoustID and LRCLIB."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


ACOUSTID_LOOKUP_URL = "https://api.acoustid.org/v2/lookup"
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = "ComfyUI-HZ3-YuE2/0.1 (audio lyrics lookup)"


def _audio_to_fingerprint(audio, fpcalc_path=""):
    """Convert ComfyUI AUDIO tensor data to a Chromaprint fingerprint."""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected a ComfyUI AUDIO input with waveform and sample_rate.")

    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().to(device="cpu", dtype=None).numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        if waveform.shape[0] < 1:
            raise ValueError("The AUDIO input contains an empty batch.")
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        raise ValueError("Expected AUDIO waveform dimensions [batch, channels, samples].")

    # ComfyUI AUDIO is channel-first. Chromaprint accepts mono or stereo WAV.
    channels_first = waveform
    if channels_first.shape[0] > 2:
        channels_first = channels_first.mean(axis=0, keepdims=True)
    pcm = channels_first.T
    sample_rate = int(audio["sample_rate"])
    if sample_rate < 1 or pcm.shape[0] < sample_rate:
        raise ValueError("The audio must contain at least one second of valid samples.")

    binary = (fpcalc_path or "").strip() or shutil.which("fpcalc")
    if not binary:
        bundled_root = Path(__file__).resolve().parent / "vendor"
        bundled_candidates = sorted(bundled_root.glob("chromaprint-*/**/fpcalc.exe"))
        if not bundled_candidates:
            bundled_candidates = sorted(bundled_root.glob("chromaprint-*/**/fpcalc"))
        if bundled_candidates:
            binary = str(bundled_candidates[-1])
    if not binary:
        raise RuntimeError(
            "Chromaprint's fpcalc was not found. Install Chromaprint (fpcalc) and add it to PATH, "
            "or provide its executable path in this node."
        )

    with tempfile.TemporaryDirectory(prefix="hz3_lyrics_") as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        sf.write(wav_path, pcm, sample_rate, subtype="PCM_16", format="WAV")
        try:
            completed = subprocess.run(
                [binary, "-json", str(wav_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"fpcalc executable was not found: {binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Chromaprint fingerprinting exceeded the 180 second limit.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "fpcalc failed").strip()
            raise RuntimeError(f"Chromaprint could not fingerprint this audio: {detail}") from exc

    try:
        result = json.loads(completed.stdout)
        duration = float(result["duration"])
        fingerprint = str(result["fingerprint"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("fpcalc returned an invalid fingerprint response.") from exc
    if duration <= 0 or not fingerprint:
        raise RuntimeError("fpcalc returned an empty fingerprint.")
    return duration, fingerprint


def _request_json(url, params, timeout=20):
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Lyrics lookup service returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the lyrics lookup service: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"Lyrics lookup request failed: {exc}") from exc


def _artist_names(recording):
    artists = recording.get("artists") or []
    return ", ".join(
        str(artist.get("name", "")).strip()
        for artist in artists
        if isinstance(artist, dict) and artist.get("name")
    )


def _best_recording(lookup, min_score):
    if lookup.get("status") != "ok":
        raise RuntimeError(f"AcoustID lookup failed: {lookup.get('error', 'unknown response')}.")
    candidates = []
    for match in lookup.get("results") or []:
        score = float(match.get("score", 0.0))
        for recording in match.get("recordings") or []:
            title = str(recording.get("title", "")).strip()
            artist = _artist_names(recording)
            if title and artist:
                candidates.append((score, recording, title, artist))
    candidates.sort(key=lambda row: row[0], reverse=True)
    for score, recording, title, artist in candidates:
        if score >= min_score:
            releases = recording.get("releasegroups") or recording.get("releases") or []
            album = ""
            if releases and isinstance(releases[0], dict):
                album = str(releases[0].get("title", "")).strip()
            return score, title, artist, album
    if candidates:
        raise RuntimeError(
            f"AcoustID found a possible match ({candidates[0][2]} by {candidates[0][3]}), "
            f"but its confidence {candidates[0][0]:.2f} is below the {min_score:.2f} threshold."
        )
    raise RuntimeError("AcoustID did not recognize this audio. This works best for released recordings.")


def _normalise_name(value):
    return " ".join("".join(char.casefold() if char.isalnum() else " " for char in value).split())


def _plain_from_lrc(synced_lyrics):
    lines = []
    for raw_line in synced_lyrics.splitlines():
        line = re.sub(r"^\s*(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]\s*)+", "", raw_line).strip()
        if line and not re.fullmatch(r"\[[a-z]{2,3}:[^]]*\]", line, flags=re.IGNORECASE):
            lines.append(line)
    return "\n".join(lines)


def _select_lyrics(search_results, title, artist, album, duration):
    if not isinstance(search_results, list) or not search_results:
        raise RuntimeError(f"No lyrics were found in LRCLIB for {title} — {artist}.")

    title_key = _normalise_name(title)
    artist_key = _normalise_name(artist)
    album_key = _normalise_name(album)
    ranked = []
    for item in search_results:
        if not isinstance(item, dict):
            continue
        candidate_title = str(item.get("trackName") or item.get("name") or "")
        candidate_artist = str(item.get("artistName") or "")
        candidate_album = str(item.get("albumName") or "")
        candidate_title_key = _normalise_name(candidate_title)
        candidate_artist_key = _normalise_name(candidate_artist)
        if not candidate_title_key or not candidate_artist_key:
            continue
        title_exact = candidate_title_key == title_key
        artist_exact = artist_key in candidate_artist_key or candidate_artist_key in artist_key
        if not title_exact or not artist_exact:
            continue
        candidate_duration = item.get("duration")
        try:
            duration_delta = abs(float(candidate_duration) - duration)
        except (TypeError, ValueError):
            duration_delta = float("inf")
        album_match = bool(album_key and album_key == _normalise_name(candidate_album))
        synced = str(item.get("syncedLyrics") or "").strip()
        plain = str(item.get("plainLyrics") or "").strip()
        if not synced and not plain:
            continue
        # Prefer title/artist/album agreement, then the nearest full-track duration,
        # then synchronized lyrics suitable for karaoke alignment.
        rank = (int(album_match), -duration_delta, int(bool(synced)))
        ranked.append((rank, item, synced, plain))
    if not ranked:
        raise RuntimeError(f"LRCLIB had no usable lyric record matching {title} — {artist}.")
    ranked.sort(key=lambda row: row[0], reverse=True)
    _, item, synced, plain = ranked[0]
    plain = plain or _plain_from_lrc(synced)
    lyrics = plain or synced
    return (
        lyrics,
        synced,
        str(item.get("trackName") or item.get("name") or title),
        str(item.get("artistName") or artist),
        str(item.get("albumName") or album),
        str(item.get("duration") or ""),
    )


class HZ3_YuE2_LyricsFromAudio:
    CATEGORY = "HZ3 YuE2/Lyrics"
    FUNCTION = "find_lyrics"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lyrics", "synced_lyrics", "title", "artist", "album", "match_score", "status")
    DESCRIPTION = (
        "Identify a released recording from AUDIO with AcoustID/Chromaprint, then fetch its lyrics "
        "from LRCLIB. The lyrics output is plain text for MixMash; synced LRC is separate. Sends "
        "an audio fingerprint rather than raw audio. Requires an AcoustID client ID and local fpcalc."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "acoustid_client_id": ("STRING", {"default": "", "multiline": False}),
                "minimum_match_score": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fpcalc_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def find_lyrics(self, audio, acoustid_client_id, minimum_match_score=0.65, fpcalc_path=""):
        client_id = str(acoustid_client_id or "").strip()
        if not client_id:
            raise ValueError("Enter your free AcoustID application client ID in the node.")
        minimum_match_score = float(minimum_match_score)
        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError("minimum_match_score must be between 0 and 1.")

        duration, fingerprint = _audio_to_fingerprint(audio, fpcalc_path)
        lookup = _request_json(
            ACOUSTID_LOOKUP_URL,
            {
                "client": client_id,
                "duration": round(duration),
                "fingerprint": fingerprint,
                "meta": "recordings+releasegroups",
            },
        )
        score, title, artist, album = _best_recording(lookup, minimum_match_score)
        search = _request_json(
            LRCLIB_SEARCH_URL,
            {"track_name": title, "artist_name": artist},
        )
        lyrics, synced, found_title, found_artist, found_album, found_duration = _select_lyrics(
            search, title, artist, album, duration
        )
        status = f"Found LRCLIB lyrics ({'synced' if synced else 'plain'}); AcoustID match {score:.2f}."
        if found_duration:
            status += f" Catalog duration: {found_duration}s."
        return lyrics, synced, found_title, found_artist, found_album, score, status


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_LyricsFromAudio": HZ3_YuE2_LyricsFromAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_LyricsFromAudio": "HZ3 YuE2 · Fetch Lyrics from Audio"}
