# kv-notary

**Can you trust a KV cache you did not compute?**

Disaggregated serving, prefix caches and KV stores (DistServe, Mooncake, vLLM, SGLang, LMCache) move a
transformer's KV cache between machines, and the node that reads it trusts it. kv-notary shows that one
forged column of the cache is enough to change an answer, and that the cache can be checked against its
own tokens for the cost of a few tokens of compute.

- **Attack.** Nudge the residual stream of one token at one layer during prefill and compute the rest
  honestly. The prompt text is unchanged, and Llama-3.2-1B answers the attacker's value in 86% of attempts.
- **Check.** Column *p* of an honest cache is what the model computes for token *p* reading columns
  0…p−1. Recomputing it against the *claimed* cache costs one token. A cache is honest if and only if
  every column passes, and a forgery fails at exactly the forged column (99.8% of cases).
- **Where to look.** Check the columns the question reads most (value-weighted attention). Eight columns
  catch 91% of forgeries; 64 random columns catch 14%.

Experiments are still finishing. Full results, the paper and the interactive demo land here shortly.

## Layout

```
kvnotary/      model (own Llama/Qwen2 forward), tasks, attacks, verifier, Notary API
experiments/   one script per experiment, figures, paper builder; run_all.sh runs everything
tests/         CPU tests on tiny random models (CI)
docs/          GitHub Pages demo, figures, paper
results/       raw JSON results
```

MIT licence.
