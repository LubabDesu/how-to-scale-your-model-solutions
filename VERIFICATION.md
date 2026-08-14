# Verification Notes

The typed solutions were checked against the corresponding chapters of
[How to Scale Your Model](https://jax-ml.github.io/scaling-book/).

## 1. Section-level notes

- Section 1 follows the Roofline exercise formulas and worked solutions.
- Section 2 follows the TPU chapter constants and derivations. Exercise 5
  retains a caveat around whether its latency term is interpreted per hop or
  across the full route.
- Section 3 Exercises 1-10 were checked against the Sharded Matmuls chapter.
  Exercises without official numerical results preserve explicit assumptions
  rather than inventing measurements.
- Section 4 Exercises 1-8 were checked against the Transformer Math chapter.
  The typed Exercise 1 corrects the scan's approximate parameter count from
  11B to approximately 16.0B.
- Section 5 Worked Problems 1-3 were checked against the training chapter.
  The typed writeup distinguishes fitting in aggregate memory from remaining
  compute-bound, which makes the pure-FSDP conclusion more precise.
- Section 6 Worked Problems 1-2 were checked against the applied-training
  chapter. The typed writeup preserves the scan's 32-33 day estimate for
  LLaMA 3-405B and makes the ideal sharding assumptions explicit.
- Section 7 Worked Problems 1-5 were checked against the inference chapter.
  The typed writeup corrects the int8 compute-time arithmetic while retaining
  the approximately 19.3 ms end-to-end estimate.
- Section 8 Worked Problems 1-3 were checked against the applied-inference
  chapter. The typed writeup separates the naive roofline result from the
  practical tensor-parallel limit.
- Section 9 Worked Problem 1 was checked against the profiling chapter. Its
  typed answer separates global shapes from local HLO shard shapes and uses an
  AllReduce for the partial down-projections.
- Section 10 implementations were checked numerically against `jax.jit`
  baselines before benchmarking on a Kaggle TPU v5e-8.
- Section 12 quizzes were checked against the GPU chapter. Ambiguous notation
  was normalized without claiming empirical GPU measurements that were not
  run.

## 2. Scan notes

- Section 2 page 1 is partially obstructed near the top. The visible exercise
  work remains legible.
- Sections 5 and 6 show a finger near the margin, but it does not cover the
  worked solutions.
- Section 12's scan includes chapter notes between quizzes. Those notes were
  excluded from the five typed quiz files.
