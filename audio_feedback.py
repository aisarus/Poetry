"""Measurements from actual PCM audio, with explicit limits on interpretation."""
from __future__ import annotations

import array
import math
import statistics
import sys
import wave
from pathlib import Path


def _samples(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), 'rb') as wav:
        if wav.getcomptype() != 'NONE' or wav.getsampwidth() not in (1, 2):
            raise ValueError(f'{path}: нужен несжатый PCM WAV 8/16-bit')
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        if rate < 8000 or channels < 1:
            raise ValueError(f'{path}: неподдерживаемая частота/каналы')
        raw = wav.readframes(wav.getnframes())
    if width == 2:
        values = array.array('h')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        normalized = [v / 32768 for v in values]
    else:
        normalized = [(v - 128) / 128 for v in raw]
    samples = [sum(normalized[i:i + channels]) / channels
               for i in range(0, len(normalized), channels)]
    return samples, rate


def _pitch_hz(frame: list[float], rate: int) -> float | None:
    stride = max(1, rate // 8000)
    reduced = frame[::stride]
    sr = rate / stride
    mean = sum(reduced) / len(reduced)
    reduced = [x - mean for x in reduced]
    candidates = []
    for lag in range(max(1, round(sr / 350)), round(sr / 70) + 1):
        a, b = reduced[:-lag], reduced[lag:]
        energy_a = sum(v * v for v in a)
        energy_b = sum(v * v for v in b)
        if not energy_a or not energy_b:
            continue
        correlation = sum(x * y for x, y in zip(a, b)) / math.sqrt(energy_a * energy_b)
        candidates.append((correlation, lag))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    if best < 0.75:
        return None
    # The highest correlation often lands on a multiple of the fundamental
    # period; prefer the earliest strong local peak to avoid a subharmonic.
    peaks = [(score, lag) for i, (score, lag) in enumerate(candidates)
             if 0 < i < len(candidates) - 1 and score >= candidates[i-1][0]
             and score >= candidates[i+1][0] and score >= max(0.75, best - 0.03)]
    _, lag = peaks[0] if peaks else max(candidates)
    return round(sr / lag, 1)


def analyze_wav(path: Path) -> dict:
    samples, rate = _samples(path)
    if not samples:
        raise ValueError(f'{path}: пустой WAV')
    hop = max(1, round(rate * 0.02))
    rms = [math.sqrt(sum(x * x for x in samples[i:i + hop]) / len(samples[i:i + hop]))
           for i in range(0, len(samples), hop)]
    threshold = max(0.004, max(rms) * 0.08)
    active = [x >= threshold for x in rms]
    speech_s = sum(active) * hop / rate
    pauses = []
    start = None
    for i, is_active in enumerate(active + [True]):
        if not is_active and start is None:
            start = i
        if is_active and start is not None:
            # Exclude leading and trailing silence; count only internal pauses.
            if start > 0 and i < len(active) and (i - start) * hop / rate >= 0.12:
                pauses.append(round((i - start) * hop / rate, 3))
            start = None
    pitches = []
    window = round(rate * 0.04)
    step = round(rate * 0.1)
    for i in range(0, len(samples) - window + 1, step):
        middle = min(len(active) - 1, int((i + window / 2) / hop))
        if active[middle]:
            value = _pitch_hz(samples[i:i + window], rate)
            if value is not None:
                pitches.append(value)
    return {'duration_s': round(len(samples) / rate, 3),
            'active_s': round(speech_s, 3),
            'internal_pauses_s': pauses,
            'pitch_median_hz_rough': round(statistics.median(pitches), 1) if pitches else None,
            'pitch_range_hz_rough': [min(pitches), max(pitches)] if len(pitches) >= 3 else None,
            'pitch_sample_count': len(pitches)}


def reading_report(lines: list[str], wav_paths: list[Path], syllables: list[int]) -> dict:
    if len(lines) != len(wav_paths) or len(lines) != len(syllables):
        raise ValueError('Число строк, аудиофайлов и слоговых оценок не совпадает')
    rows = []
    for n, (line, path, count) in enumerate(zip(lines, wav_paths, syllables), 1):
        measures = analyze_wav(path)
        active_s = measures['active_s']
        rows.append({'line': n, 'text': line, 'wav': str(path), **measures,
                     'syllables_per_active_s_approx': round(count / active_s, 2) if active_s else None})
    return {'source': 'synthesized_pcm_wav', 'lines': rows,
            'limits': 'Измерены длительность, энергия, паузы и грубая периодичность. Это не распознавание слов, слоговых акцентов или художественная оценка. Синтетический голос может произнести ударения неверно.'}
