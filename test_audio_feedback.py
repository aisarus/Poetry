import contextlib
import io
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import audio_feedback
import poetry_lab


def tone_wav(path: Path, hz: int = 220) -> None:
    rate = 16000
    segments = [(0.25, True), (0.20, False), (0.25, True)]
    data = []
    for duration, voiced in segments:
        for i in range(round(duration * rate)):
            data.append(round(12000 * math.sin(2 * math.pi * hz * i / rate)) if voiced else 0)
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack('<' + 'h' * len(data), *data))


class AudioFeedbackTests(unittest.TestCase):
    def test_waveform_pause_and_pitch_are_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tone.wav'
            tone_wav(path)
            result = audio_feedback.analyze_wav(path)
            self.assertAlmostEqual(result['duration_s'], .7, delta=.01)
            self.assertTrue(any(.15 <= p <= .24 for p in result['internal_pauses_s']))
            self.assertAlmostEqual(result['pitch_median_hz_rough'], 220, delta=10)

    def test_critic_receives_real_waveform_metrics_before_revision(self):
        poem = 'На крыше дома репетировали май\nИ окна разливались акварелью\nТрамваи пахли свежей карамелью\nНе прозевай!'
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp) / 'input.txt'
            text.write_text(poem, encoding='utf-8')
            order = []

            def synth(current, run_dir, step):
                order.append(('synth', step))
                full = run_dir / f'reading_{step:02d}.wav'
                tone_wav(full)
                paths = []
                for n, _ in enumerate(current.splitlines(), 1):
                    path = run_dir / f'line_{step}_{n}.wav'
                    tone_wav(path)
                    paths.append(path)
                return full, paths

            def chat(messages, *args):
                order.append(('chat', len(order)))
                if len([x for x in order if x[0] == 'chat']) == 1:
                    self.assertIn('"source": "synthesized_pcm_wav"', messages[1]['content'])
                    self.assertIn('"internal_pauses_s": [0.18]', messages[1]['content'])
                    return 'Пауза в каждой строке; проверить чтение.'
                return poem

            with patch.object(poetry_lab, 'speak_windows_batch', side_effect=synth), \
                 patch.object(poetry_lab, 'ollama_chat', side_effect=chat), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = poetry_lab.main(['--file', str(text), '--rounds', '1',
                                        '--audio-feedback', '--out', tmp])
            self.assertEqual(code, 0)
            self.assertEqual(order[0], ('synth', 0))
            self.assertEqual(order[-1], ('synth', 1))
            run_dir = next(p for p in Path(tmp).iterdir() if p.is_dir())
            trace = json.loads((run_dir / 'trace.json').read_text(encoding='utf-8'))
            self.assertIn('audio_feedback', trace[0]['prosody_report'])
            self.assertIn('audio_feedback', trace[1]['prosody_report'])
            self.assertTrue((run_dir / 'audio_metrics_00.json').exists())


if __name__ == '__main__':
    unittest.main()
