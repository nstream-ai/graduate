# References

The methods Graduate implements, and the work it builds on.

## Statistics of the certificate
- Clopper, C. J. & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in
  the case of the binomial. *Biometrika* 26(4).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *JRSS-B* 57(1).
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple
  testing under dependency. *Annals of Statistics* 29(4).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal
  of Statistics* 6(2).
- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling.
  *Technometrics* 24(4).
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions
  or percentages. *Psychometrika* 12(2).
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Waudby-Smith, I. & Ramdas, A. (2024). Estimating means of bounded random variables by betting.
  *JRSS-B*. arXiv:2010.09686.
- Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023). Game-theoretic statistics and safe
  anytime-valid inference. *Statistical Science*. arXiv:2210.01948.

## Statistics for evaluating models
- Miller, E. (2024). Adding error bars to evals: a statistical approach to language model
  evaluations. arXiv:2411.00640.
- Bowyer, S., Aitchison, L. & Ivanova, D. R. (2025). Don't use the CLT in LLM evals with fewer than
  a few hundred datapoints. ICML. arXiv:2503.01747.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification
  learning algorithms. *Neural Computation* 10(7).
- Madaan, L. et al. (2024). Quantifying variance in evaluation benchmarks. arXiv:2406.10229.

## Selective prediction and risk control
- Geifman, Y. & El-Yaniv, R. (2017). Selective classification for deep neural networks. NeurIPS.
  arXiv:1705.08500.
- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I. & Lei, L. (2021). Learn then test:
  calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
- Angelopoulos, A. N. et al. (2024). Conformal risk control. ICLR. arXiv:2208.02814.
- Jung, J., Brahman, F. & Choi, Y. (2025). Trust or escalate: LLM judges with provable guarantees
  for human agreement. ICLR. arXiv:2407.18370.

## Agents and benchmarks
- Yao, S., Shinn, N., Razavi, P. & Narasimhan, K. (2024). τ-bench: a benchmark for tool-agent-user
  interaction in real-world domains. arXiv:2406.12045.
- Barres, V., Dong, H., Ray, S., Si, X. & Narasimhan, K. (2025). τ²-bench: evaluating
  conversational agents in a dual-control environment. arXiv:2506.07982.
- Stanton, S. et al. (2021). Does knowledge distillation really work? NeurIPS. arXiv:2106.05945.
- Panickssery, A., Bowman, S. R. & Feng, S. (2024). LLM evaluators recognize and favor their own
  generations. NeurIPS. arXiv:2404.13076.

## Trace formats and data sources (ingest presets)
- NousResearch (2024). hermes-function-calling-v1 (dataset, Apache-2.0).
  huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- Xu, Z. et al. (2025). TOUCAN: synthesizing 1.5M tool-agentic data from real-world MCP
  environments. arXiv:2510.01179. Dataset: huggingface.co/datasets/Agent-Ark/Toucan-1.5M (Apache-2.0).
- NVIDIA (2025). Nemotron-Agentic-v1 (dataset, CC-BY-4.0). huggingface.co/datasets/nvidia/Nemotron-Agentic-v1
- Badertdinov, I. et al. (2025). SWE-rebench: an automated pipeline for task collection and
  decontaminated evaluation of software engineering agents. arXiv:2505.20411.
  Dataset: huggingface.co/datasets/nebius/SWE-rebench (CC-BY-4.0).
- Sierra Research. τ²-bench simulation format. github.com/sierra-research/tau2-bench

## Models and chat formats named in the docs and the training code
- Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Yang, A. et al. (2025). Qwen3 technical report. arXiv:2505.09388.
- Google DeepMind (2026). Gemma 4 (Apache-2.0). ai.google.dev/gemma. The chat-template test fixture
  is Google's published template; see `NOTICE`.
- Comanici, G. et al. (2025). Gemini 2.5: pushing the frontier with advanced reasoning,
  multimodality, long context, and next generation agentic capabilities. arXiv:2507.06261.
- Hu, E. J. et al. (2021). LoRA: low-rank adaptation of large language models. arXiv:2106.09685.
- Kwon, W. et al. (2023). Efficient memory management for large language model serving with
  PagedAttention (vLLM). SOSP. arXiv:2309.06180.

## Practice
- Karpathy, A. (2025). Public posts on the "evaluation crisis": private, rotating evaluation sets
  (the motivation for `core.split.rolling_rounds`).
