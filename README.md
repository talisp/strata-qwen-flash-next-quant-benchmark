# Qwen3.8-Flash-Next at 4, 6 and 8 bits on a 48 GB vGPU: speed from 256K to 1M context (Strata v0.1.42)

I ran the same performance battery on three quantizations of `unsloth/Qwen3.8-Flash-Next-GGUF` with Strata v0.1.42, on one NVIDIA L40S-48Q vGPU. It covers 18 configurations: 3 weight formats × 3 context sizes (262,144 / 524,288 / 1,048,576) × 2 KV cache types (int8 and fp16).

**Why we run this.** This model serves a local agent that works inside Unreal Engine through MCP (field report in [Strata #980](https://github.com/Niko1221/Strata/issues/980)). The sessions are long, often 100K–190K tokens of context, and include viewport captures. So prompt reading speed at large context and vision matter more to us than short-prompt speed. This report follows [Strata #971](https://github.com/Niko1221/Strata/issues/971), where I promised to test larger quantizations.

## Results in short

Tokens per second, KV int8:

| weights | reading, 32K prompt | generation, 32K prompt | reading, full 1M | generation, full 1M |
|---|---:|---:|---:|---:|
| UD-IQ4_XS (4-bit) | 4,210 | 136 | 2,670 | 74 |
| UD-Q6_K_XL (6-bit) | 2,225 | 102 | 1,683 | 60 |
| Q8_0 (8-bit) | 1,840 | 104 | 1,459 | 58 |

- **All 18 configurations worked, including 1M context.** In every run the model returned the 3 codes hidden in the full context, in order.
- **UD-IQ4_XS reads prompts about 2× faster** than the other two and generates about 30% faster.
- **UD-Q6_K_XL and Q8_0 are close.** Q6 reads 15–20% faster; generation is a tie.
- **KV fp16 makes reading slower and leaves generation unchanged.** Reading a full context was 15–16% slower on UD-IQ4_XS, 10–12% on UD-Q6_K_XL and 8–11% on Q8_0. It also doubles the RAM the KV cache takes.
- **A larger context costs speed only when it is actually full.** From 256K to 1M, reading a full context got 17–31% slower and generation 14–19% slower. With a 32K prompt, the configured context size made almost no difference.
- **Strata itself needed no fix.** No crash or engine error in any run. UD-Q6_K_XL needs an engine built with two flags, and both UD-Q6_K_XL and Q8_0 need a hand-written config (details below).

## Which one to pick

What each quantization needs, on this machine:

| | UD-IQ4_XS (4-bit) | UD-Q6_K_XL (6-bit) | Q8_0 (8-bit) |
|---|---:|---:|---:|
| Files on disk | 93.7 GB | 169.2 GB | 188.2 GB |
| Experts loaded into RAM | 55.4 GiB | 101.8 GiB | 119.5 GiB |
| Experts that fit on the card (~33 GiB) | 14,613 | 7,886 | 6,712 |
| Startup | 23 s – 2.3 min | 39 s – 9.7 min | 1.4 – 10 min |
| Reading, full 256K context (tok/s) | 3,844 | 2,112 | 1,760 |
| Generation, full 256K context (tok/s) | 91 | 70 | 69 |
| Works with the standard setup | yes | no (custom build + config) | no (config by hand) |

- **For speed, UD-IQ4_XS.** It is the only one of the three that the setup installs, it needs half the RAM, and it is the fastest at every context size.
- **Between the two larger ones, UD-Q6_K_XL** is smaller and a little faster than Q8_0, but it is the one that needs a custom engine build.
- **KV cache: int8.** fp16 costs reading speed and RAM, and we measured no gain from it (including on vision, see below).
- **Context: set it as large as you need.** An unused part of the context costs almost no speed. It does cost RAM for the KV cache (3 GiB at 256K, 12 GiB at 1M, with int8).
- **This is a speed test.** It does not tell you whether the 6-bit or 8-bit weights give better answers for your task.

## Hardware

| | |
|---|---|
| GPU | NVIDIA L40S-48Q vGPU (VMware): 48 GB reported, about 42.2 GiB allocatable |
| Driver | 550.127.05 (cannot be upgraded from inside the VM); CUDA 12.9 runtime in a user folder |
| CPU | Intel Xeon Platinum 8580, 32 vCPU, AVX-512 |
| RAM | 251 GiB |
| Disk | 4 TB virtual disk on SSD |
| OS | Ubuntu 22.04, no root access |

What mattered for Strata on this machine:

- **VRAM sets the speed.** The card holds about 33 GiB of experts whatever the quantization, so smaller weights mean more experts on the card.
- **RAM sets what can run at all.** The experts are loaded into RAM: 55 GiB at 4 bits, 102 GiB at 6 bits, 120 GiB at 8 bits, plus the KV cache. A machine with less RAM than that would need Strata's low-RAM mode, which I did not test here.
- **The SSD matters at startup** (reading 94–188 GB of files) and for the 54.4 GB PLE table, which is read from disk.
- **A vGPU has less VRAM than it reports** and a fixed driver. Details in [Strata #971](https://github.com/Niko1221/Strata/issues/971).

<details>
<summary><b>All 18 runs with the context full</b></summary>

One run per row. The prompt is a long document with 3 codes hidden in it; the model has to return them in order. Generation speed is measured on a follow-up question that reuses the cached context.

| weights | KV | context | prompt tokens | reading (tok/s) | time to read it all | generation (tok/s) |
|---|---|---|---:|---:|---:|---:|
| UD-IQ4_XS | int8 | 256K | 250,616 | 3,844 | 1.1 min | 91 |
| UD-IQ4_XS | int8 | 512K | 497,579 | 3,448 | 2.4 min | 83 |
| UD-IQ4_XS | int8 | 1M | 1,041,873 | 2,670 | 6.5 min | 74 |
| UD-IQ4_XS | fp16 | 256K | 250,615 | 3,249 | 1.3 min | 92 |
| UD-IQ4_XS | fp16 | 512K | 497,579 | 2,888 | 2.9 min | 87 |
| UD-IQ4_XS | fp16 | 1M | 1,041,872 | 2,230 | 7.8 min | 76 |
| UD-Q6_K_XL | int8 | 256K | 250,616 | 2,112 | 2.0 min | 70 |
| UD-Q6_K_XL | int8 | 512K | 497,579 | 1,991 | 4.2 min | 68 |
| UD-Q6_K_XL | int8 | 1M | 1,041,872 | 1,683 | 10.3 min | 60 |
| UD-Q6_K_XL | fp16 | 256K | 250,615 | 1,910 | 2.2 min | 71 |
| UD-Q6_K_XL | fp16 | 512K | 497,579 | 1,767 | 4.7 min | 65 |
| UD-Q6_K_XL | fp16 | 1M | 1,041,872 | 1,478 | 11.8 min | 58 |
| Q8_0 | int8 | 256K | 250,616 | 1,760 | 2.4 min | 69 |
| Q8_0 | int8 | 512K | 497,578 | 1,671 | 5.0 min | 66 |
| Q8_0 | int8 | 1M | 1,041,873 | 1,459 | 11.9 min | 58 |
| Q8_0 | fp16 | 256K | 250,615 | 1,616 | 2.6 min | 70 |
| Q8_0 | fp16 | 512K | 497,578 | 1,518 | 5.5 min | 64 |
| Q8_0 | fp16 | 1M | 1,041,871 | 1,302 | 13.3 min | 57 |

"Time to read it all" is prompt tokens ÷ reading speed.

</details>

<details>
<summary><b>Speed with shorter prompts (4K, 32K, 128K)</b></summary>

Median of 3 runs, 256 generated tokens. Reading and generation in tokens per second.

| weights | KV | context | 4K reading | 4K generation | 32K reading | 32K generation | 128K reading | 128K generation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| UD-IQ4_XS | int8 | 256K | 2,422 | 130 | 4,210 | 136 | 4,033 | 119 |
| UD-IQ4_XS | int8 | 512K | 2,341 | 142 | 4,166 | 136 | 4,014 | 127 |
| UD-IQ4_XS | int8 | 1M | 2,161 | 124 | 4,108 | 142 | 3,952 | 117 |
| UD-IQ4_XS | fp16 | 256K | 2,342 | 137 | 3,566 | 135 | 3,425 | 121 |
| UD-IQ4_XS | fp16 | 512K | 2,242 | 129 | 3,563 | 142 | 3,412 | 115 |
| UD-IQ4_XS | fp16 | 1M | 2,064 | 132 | 3,503 | 136 | 3,358 | 121 |
| UD-Q6_K_XL | int8 | 256K | 964 | 103 | 2,225 | 102 | 2,162 | 96 |
| UD-Q6_K_XL | int8 | 512K | 952 | 104 | 2,211 | 106 | 2,146 | 103 |
| UD-Q6_K_XL | int8 | 1M | 924 | 96 | 2,146 | 105 | 2,101 | 99 |
| UD-Q6_K_XL | fp16 | 256K | 942 | 105 | 2,066 | 108 | 1,982 | 98 |
| UD-Q6_K_XL | fp16 | 512K | 933 | 101 | 2,037 | 95 | 1,954 | 92 |
| UD-Q6_K_XL | fp16 | 1M | 890 | 95 | 1,989 | 103 | 1,910 | 95 |
| Q8_0 | int8 | 256K | 787 | 95 | 1,840 | 104 | 1,798 | 91 |
| Q8_0 | int8 | 512K | 778 | 92 | 1,817 | 94 | 1,779 | 89 |
| Q8_0 | int8 | 1M | 764 | 89 | 1,802 | 88 | 1,762 | 86 |
| Q8_0 | fp16 | 256K | 778 | 99 | 1,734 | 95 | 1,667 | 89 |
| Q8_0 | fp16 | 512K | 767 | 98 | 1,711 | 92 | 1,651 | 88 |
| Q8_0 | fp16 | 1M | 746 | 94 | 1,681 | 86 | 1,626 | 95 |

</details>

<details>
<summary><b>Where the memory went (VRAM, RAM, SSD)</b></summary>

- **VRAM:** about 43.5 GiB of the 48 GiB reported by the vGPU was in use in all 18 runs. Strata fills the card with as many experts as fit: 14,613 for UD-IQ4_XS, 7,886 for UD-Q6_K_XL and 6,712 for Q8_0 (about 33 GiB each, at 256K with KV int8). More experts on the card is what makes the 4-bit file faster.
- **KV cache:** it lives in pinned RAM. Only a window of 32,768 cells per layer stays in VRAM, whatever the context size (`--kv-resident 32768`).

  | context | KV int8 in RAM | KV fp16 in RAM |
  |---|---:|---:|
  | 262,144 | 3.09 GiB | 6.00 GiB |
  | 524,288 | 6.19 GiB | 12.00 GiB |
  | 1,048,576 | 12.38 GiB | 24.00 GiB |

  Even so, 75–86% of the KV block reads were served from VRAM.
- **RAM:** the experts are loaded into RAM: 101.75 GiB for UD-Q6_K_XL and 119.53 GiB for Q8_0. System RAM in use at the end of a run was about 62 GiB (UD-IQ4_XS), 110 GiB (UD-Q6_K_XL) and 128 GiB (Q8_0).
- **SSD:** the 54.4 GB PLE table is read from disk (see the setup notes). The KV cache never goes to disk.

</details>

<details>
<summary><b>What it took to run UD-Q6_K_XL and Q8_0</b></summary>

UD-IQ4_XS is what the setup installs, so it just works. The other two needed manual steps:

1. **Engine build (UD-Q6_K_XL only).** Build from source with `-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON`. Both are off in the default build and in the prebuilt engine. Q8_0 runs on the default build.
2. **Pack.** `tools/iq_pack.py --gguf <shard 1> --out packs/<name> --compat-bf16`, about 40 s per model. No `--experts-bin`; the experts stay in the GGUF files.
3. **Config by hand.** The setup refuses these GGUF files. What was not obvious:
   - KV fp16 is `--kv fp16` (the setup only offers int8, q4_0 and k8v4).
   - 524,288 context needs `--rope-scaling yarn --rope-scale 2`; 1,048,576 needs `--rope-scale 4`.
   - Vision with another GGUF: `"vision": {"model": <shard 1 of that model>}`.
   - Do not use `--ple-io ram` when `ulimit -l` is low (31.5 GiB on this VM): the PLE table is 54.4 GB.
4. **Disk.** Shards 2 and 3 of UD-Q6_K_XL and Q8_0 are byte-identical (same SHA-256), so a hardlink saves 55 GB.

Startup time depends on whether the files are already in the page cache: 23 s to 2.3 min for UD-IQ4_XS, 39 s to 9.7 min for UD-Q6_K_XL, 1.4 to 10 min for Q8_0.

</details>

<details>
<summary><b>Does KV fp16 help vision?</b></summary>

Not measurably here. I asked 24 questions about synthetic images with known answers (small text at 9, 12, 16 and 24 px, counting dots, reading a table cell, naming the colour at a grid position), at 256K context:

| weights | KV int8 | KV fp16 |
|---|---:|---:|
| UD-IQ4_XS | 17 / 24 | 18 / 24 |
| UD-Q6_K_XL | 17 / 24 | 17 / 24 |
| Q8_0 | 17 / 24 | 17 / 24 |

The three models miss the same things: the 9 px text (usually one wrong character) and the dot counts (off by one). Tables and colours were always right. A simpler check (name 3 shapes and their colours) passed in all 18 configurations.

</details>

<details>
<summary><b>History: our Strata tests on this machine, from the first install to now</b></summary>

Same VM and card throughout. The speeds below were measured in different ways (agent sessions, a single check after an upgrade, this battery), so read them as a log, not as a controlled comparison between versions.

| date | Strata | model | what happened | reading (tok/s) | generation (tok/s) |
|---|---|---|---|---:|---:|
| 2026-10-01 | 0.1.38 | IQ3_S, no images | First install. Driver 550 was refused; ran with a user-folder CUDA 12.4 and a wrapper. About 20 h of agent work without vision. | ~3,980 | ~96 |
| 2026-10-04 | 0.1.38 | IQ3_S + images | Vision turned on. Found the tool call written inside `<think>` ([Strata #804](https://github.com/Niko1221/Strata/issues/804)). | ~3,920 | ~98 |
| 2026-10-05 | 0.1.39 | UD-IQ4_XS + images | `--cuda 12` accepts driver 550 officially. First report that UD-IQ4_XS works with images ([Strata #971](https://github.com/Niko1221/Strata/issues/971)). Unreal field report ([Strata #980](https://github.com/Niko1221/Strata/issues/980)). | ~3,200–3,300 | ~77–84 |
| 2026-10-06 | 0.1.40.1 | UD-IQ4_XS + images | The 0.1.40 engine does not build with CUDA 12.4; moved to CUDA 12.9 (still driver 550). The `<think>` fix is now upstream. Setup removed `api_key` and `host` from the config; restored by hand. | 3,321 | 83 |
| 2026-10-10 | 0.1.42 | UD-IQ4_XS, UD-Q6_K_XL, Q8_0 | This report: 18 configurations, 256K to 1M context, KV int8 and fp16. | 4,033 | 119 |

The first four rows are at roughly 90K tokens of context in real agent use or a single check. The last row is UD-IQ4_XS with a 128K prompt, KV int8, thinking off (median of 3).

Lessons that carried over between versions:

- Keep each version's engine and config in its own folder, so a rollback is a restart.
- After every setup run, check `api_key`, `host`, sampling and aliases in the config.
- Add `"aliases": ["<old model name>"]` so the client does not need to be reconfigured.
- On a vGPU, plan VRAM against what is allocatable, not what is reported.

</details>

<details>
<summary><b>Method and caveats</b></summary>

- **Hardware:** NVIDIA L40S-48Q vGPU (about 42 GiB usable), driver 550, CUDA 12.9 runtime, Xeon Platinum 8580 (32 cores), 251 GiB RAM, SSD, Ubuntu 22.04.
- **Software:** Strata v0.1.42 built from source (sm_89). Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `38bb39e`.
- **Same settings for the three models.** The configs differ only in the model files. Engine arguments: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --kv-resident 32768 --vision --vram-reserve-mib 700`.
- **Requests:** temperature 0, thinking off, one request at a time.
- **Full-context numbers are a single run each.** Generation speed varies by about ±10% between runs, so small differences in generation mean nothing.
- **Not included:** cache hit-rate counters. A bug in my measuring script read the wrong request for part of the battery, so I left them out. Speeds come from the per-request timings and are not affected.
- **Scope:** this is a speed test. I did not try to rank the answer quality of the three quantizations.

</details>

## Data

The numbers behind every table are in [`data/`](data):

- [`full-context.csv`](data/full-context.csv): the 18 runs with the context full.
- [`short-prompts.csv`](data/short-prompts.csv): every run of the 4K / 32K / 128K prompts (3 runs each).
- [`vision-detail.csv`](data/vision-detail.csv): the 24 vision questions per configuration, with the expected and the given answer.

Published as ISSUE_LINK in the Strata repository.

## Resumo em português (pt-BR)

Teste de velocidade de três quantizações do Qwen3.8-Flash-Next (UD-IQ4_XS de 4 bits, UD-Q6_K_XL de 6 bits e Q8_0 de 8 bits) no Strata v0.1.42, numa vGPU NVIDIA L40S-48Q, com contexto de 256K, 512K e 1M e KV cache int8 e fp16 (18 configurações).

- **As 18 configurações funcionaram, inclusive com 1M de contexto.** Em todas, o modelo achou os 3 códigos escondidos no contexto cheio.
- **O de 4 bits é o mais rápido:** lê o prompt cerca de 2× mais rápido que os outros dois e gera cerca de 30% mais rápido. Também usa metade da RAM (55 GiB contra 102 e 120 GiB) e é o único que o setup instala sozinho.
- **O de 6 e o de 8 bits ficam próximos:** o de 6 bits lê 15–20% mais rápido; na geração dá empate.
- **KV fp16 não compensou:** a leitura fica 8–16% mais lenta, a RAM do KV dobra e não medimos ganho, nem na visão.
- **Contexto maior só custa velocidade quando está cheio.** Deixar o contexto configurado em 1M quase não muda a velocidade de um prompt de 32K; custa RAM para o KV (12 GiB em 1M com int8).
- **O Strata não precisou de correção.** O de 6 bits exige compilar o motor com duas opções, e o de 6 e o de 8 bits exigem configuração feita à mão.
- **Este é um teste de velocidade.** Ele não diz qual quantização responde melhor.
