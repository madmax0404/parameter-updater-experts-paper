# Parameter Updater Experts

Preprint and supporting materials for:

**Parameter Updater Experts: Inference-Time Learning in MoE Models via DeltaNet-LoRA**
Jongyun (Max) Han — 맥스와옴니스 주식회사 / Max & Omnis Inc.
April 2026

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19661389-blue)](https://doi.org/10.5281/zenodo.19661389)
[![SSRN](https://img.shields.io/badge/SSRN-6555479-orange)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6555479)
[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

*The Zenodo record and this repository hold the current version (v7). The SSRN listing is an earlier draft.*

## What this is

An architectural proposal for a new role within Mixture-of-Experts layers. Some expert slots are designated not as inference computations but as **parameter updater experts** whose output is a weight delta ($\Delta w$) applied to sibling inference experts. Under this concept the MoE router's sparse gating mechanism implicitly determines both what to compute *and* when and whether the model adapts its own parameters.

The preprint also presents **DeltaNet-LoRA**, a mechanism-isolation prototype that validates a necessary submechanism of the full framework — learned, persistent weight-delta generation — on a real pretrained MoE (OLMoE-1B-7B). DeltaNet-LoRA is a constrained retrofit experiment, not a full instantiation of the framework; router-selected updater activation, expert-indexed peer modification, and multi-updater interactions remain future work.

## Highlights from the preprint

- Four-fold novelty mapping against prior art at the intersection of HyperNetworks, Fast Weight Programmers, and Mixture of Experts (§2).
- Framework definition with expert role taxonomy, $\Delta w$ parameterization design space, router semantics, multi-updater extensions, and persistent state (§3).
- On OLMoE-1B-7B with all base weights frozen, DeltaNet-LoRA reaches 80.1% persistent fact retrieval under a sliding-window attention constraint (mechanism validation), 52.2% under full causal attention via a dual-updater variant, and 54.0% on natural-language templated facts with a single persistent updater drawing from per-layer hidden states (§5).
- Ablations showing simple gated linear interpolation beats surprise-gating and delta-rule variants at low rank on a single-researcher compute budget.
- A parallel-scan reformulation yielding 2.77× training speedup with comparable accuracy.

## Repository contents

| Path | Contents |
|---|---|
| `preprint/` | The preprint PDF (`preprint-v7.pdf`). |
| `repro/` | Standalone reproduction bundle for the DeltaNet-LoRA parallel-scan equivalence check and recurrence microbenchmark. |
| `idea/` | Earlier idea-stage documents, including the prior-art novelty analysis (`updater_expert.pdf`). |
| `prior-arts/` | PDFs of the academic works and patents referenced in the preprint. |
| `LICENSE.md` | License terms (CC BY-NC-ND 4.0). |

## Citation

Please cite the Zenodo record (current version):

```bibtex
@misc{han2026parameter,
  author       = {Han, Jongyun},
  title        = {Parameter Updater Experts: Inference-Time Learning
                  in MoE Models via DeltaNet-LoRA},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19661389},
  url          = {https://zenodo.org/records/19661389}
}
```

The paper is also indexed on SSRN (earlier draft): <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6555479>.

## External citations

This preprint has been cited as Han (2026) in Liu et al. from Peking University, **SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass**, which was accepted by ICML 2026, arXiv:2602.06358v2 (May 20, 2026): <https://arxiv.org/html/2602.06358v2>.

## Reproduction Bundle

This repository includes a small, standalone reproduction bundle under `repro/`.
It verifies the gated-lerp parallel-scan algebra used by DeltaNet-LoRA and provides a recurrence-only microbenchmark. It does not publish the full private OLMoE training workspace or reproduce the full experiment-28 training run.

Quick check:

```bash
python repro/parallel_scan_equivalence.py
```

For PyTorch installation via this repo metadata:

```bash
uv sync --extra repro
```

The full DeltaNet-LoRA training code and experiment scripts are maintained privately by the author and may be made available on request.

## License

This preprint and its supporting materials are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). You are free to share the material with attribution, but commercial use, derivative works, and adaptations require permission.

Contact: `madmax0404@maxandomnis.com`
