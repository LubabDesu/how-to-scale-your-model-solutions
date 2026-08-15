# Section 10: Programming TPUs in JAX

These exercises were run on a Kaggle TPU v5e-8.

## 1. Notebooks

| Problems | Notebook | Coverage |
| --- | --- | --- |
| Question 1 | [Shard averaging](https://www.kaggle.com/code/lucasyanyk/scaling-chatper10-question-1) | `shard_map`, `jax.jit`, communication inspection, timing |
| Questions 2-3 | [MoE and collective matmuls](https://www.kaggle.com/code/lucasyanyk/chapter10-scalingbookq2q3) | MoE routing, `all_to_all`, top-k routing, AllReduce and ReduceScatter matmuls, Transformer block |
| Question 4 | [Bidirectional collectives](https://www.kaggle.com/code/lucasyanyk/chatper10-scalingbookq4-bidirectional) | Bidirectional AllReduce and ReduceScatter using `ppermute` |

## 2. Benchmark results

Question 1 produced matching shard averages:

| Implementation | Median time |
| --- | ---: |
| `shard_map` | 546.97 us |
| `jax.jit` | 545.65 us |

Question 2's `jax.jit` MoE baseline used an AllGather and measured 614.54 us
median, with a 570.09 us minimum. The explicit, ragged `all_to_all`, and top-k
implementations passed their correctness checks.

Question 3's full collective Transformer block measured 640.39 us versus
653.81 us for `jax.jit`, giving a 1.021x measured speedup.

Question 4 used a `2 x 4` mesh:

| Operation | Implementation | Median time |
| --- | --- | ---: |
| AllReduce matmul | `jax.jit` | 1136.42 us |
| AllReduce matmul | tiled `psum` | 1156.62 us |
| AllReduce matmul | bidirectional | 1430.30 us |
| ReduceScatter matmul | `jax.jit` | 617.97 us |
| ReduceScatter matmul | unidirectional ring | 643.16 us |
| ReduceScatter matmul | bidirectional | 632.61 us |

The bidirectional ReduceScatter was 1.017x faster than the unidirectional ring
in this run. The manual bidirectional AllReduce was slower than both the tiled
`psum` and compiler-generated baselines, showing that reduced communication
volume alone did not guarantee a faster XLA schedule.
