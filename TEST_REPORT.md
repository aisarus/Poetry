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

## Update — 2026-11-09

The original 8 checks still pass. Six additional tests cover stress-aware rhyme hints, unknown accents, a PCM WAV with a known 220 Hz tone and internal pause, and an end-to-end critic/revision cycle where actual waveform measurements enter the critic prompt and trace. `python -m unittest discover -q`: 14/14 PASS in this Linux environment.

The audio orchestration used a simulated synthesizer that wrote real WAV bytes. Windows SAPI itself and a real Ollama model remain untested here. The pitch estimator is rough and the phonetic layer lacks full Russian G2P. Neither technical check establishes artistic quality.

## Masked diffusion prototype — 2026-11-09

`masked_diffusion.py` implements random-rate absorbing-mask training on accepted draft/revision pairs and confidence-ranked iterative unmasking at inference. A Russian pretrained MLM checkpoint is configured but has **not been downloaded or fine-tuned** in this environment: PyTorch and Transformers are absent. Three pure Python tests verify corruption/labels, line and stanza boundaries, locked lines, reverse commitment schedule, and human acceptance gating for trace exports. Total: 17/17 tests pass. CLI parsing and bytecode compilation pass. This does not establish training convergence or good poetic output.

## Training and blind evaluation preparation — 2026-11-09

The current environment has no NVIDIA GPU or PyTorch/Transformers. Direct HTTPS requests to PyPI and Hugging Face timed out; no pretrained weights or accepted revision dataset are present. No fine-tuning or human preference study has run. Added a tiny **real** PyTorch backward/optimizer/reverse-sampling smoke test for CI, where dependencies can be installed. Added `blind_eval.py` to randomize A/B candidates with a hidden key, reject incomplete ballots, and compute wins/ties and a Wilson interval. Here: 19 discovered tests, 18 pass, one PyTorch smoke test skipped; bytecode compilation passes.

GitHub Actions [run 42542742339](https://github.com/aisarus/Poetry/actions/runs/42542742339) completed successfully after installing CPU PyTorch and Transformers; its tiny-model test step passed. It does **not** verify pretrained Russian weights, full fine-tuning, or literary quality.

## Public-domain bootstrap preparation — 2026-11-09

Added a Wikisource collector restricted to Pushkin category pages with `PD-old-70` tag, source URLs retained per passage, and a Colab notebook for GPU bootstrap training. Added an end-to-end tiny local Transformer train/sample test for CI and a live corpus-fetch smoke step. Here: 23 discovered tests, 21 pass, 2 PyTorch tests skipped. Colab sign-in was not completed, so no actual GPU training or generated poem from the pretrained Russian model exists yet. The next CI run must validate the new end-to-end and external fetch steps; source HTML/API can change.

## Verified integration — 2026-11-09

[GitHub Actions run 42545815583](https://github.com/aisarus/Poetry/actions/runs/42545815583) passed 23/23 tests with CPU PyTorch and Transformers installed. This includes saving and sampling a trained tiny BERT checkpoint. The live Wikisource step found the `PD-old-70` hidden category on one named Pushkin page and extracted six passages. This validates the collector on that page, but not the planned 150-page corpus. Local environment: 21 passed, two dependency-gated tests skipped. The pretrained `ai-forever/ruBert-base` model has not yet been trained or sampled: Google Colab requires account sign-in.
