# Poetry Lab MVP — smoke-test report

Environment: Linux container, Python 3.13.5. No Ollama process, no locally installed models, and no Windows SAPI available in this environment.

## Checks

- 8/8 Python `unittest` tests: PASS.
- Python bytecode compilation of application and tests: PASS.
- CLI argument parser (`--help`): PASS.
- Draft → critique → revision workflow with simulated model replies: PASS.
- Persistence of `trace.json`, per-stage poems/metrics/critiques, and final output: PASS.
- `/api/chat` JSON HTTP protocol via loopback stub server, including Cyrillic UTF-8 and HTTP 503 handling: PASS.
- Static text analysis without Ollama: PASS.

## What these checks do not prove

- No full generation with a **real** Ollama model was run, because no Ollama service or model is present.
- No WAV audio was generated, because Windows SAPI is unavailable on Linux.
- Real-time prosody analysis, listening feedback, accurate stress-aware meter/rhyme checking, and diffusion modeling are **not implemented** in this MVP.
- No objective artistic-quality test was run; the simulated model replies are fixed fixtures.

## Reproduce on Windows

```powershell
python -m unittest discover -v
ollama pull qwen2.5:7b
python poetry_lab.py --model qwen2.5:7b --topic "Праздник маленькой победы" --stanzas 5 --rounds 1 --speak
```

`qwen2.5:7b` runs in the user's own Ollama environment; it does **not** consume ChatGPT Work or GPT model message allowances. Start with `--rounds 1` to minimize local runtime. To run the ChatGPT task in Work mode or set the ChatGPT reasoning effort to Medium, choose those options in the ChatGPT interface; these modes cannot be switched by this Python program.
