# Checkpoint and exit at the end of a data tranche

Add the following to the size file's `EXTRA_NETWORK_ARGS`:

```bash
--exit-on-tranche-end
```

The configured training data paths define the current tranche. The GPT dataset
builder creates a single pass over each contributing shard, with exhaustive
blending rather than repeating data to fill `--train-samples`. Explicit blend
weights are unsupported; pass unweighted shard prefixes, as the megachonk
launcher does when `DATA_SOURCES="."`.

The full-run training budget and LR decay are unchanged. All pipeline/tensor
parallel ranks receive the same sample boundary. At the last complete global
batch, the run writes a persistent checkpoint, finalizes any asynchronous save,
and exits successfully. The boundary save takes precedence over interval saves,
including non-persistent intervals.

Samples that cannot fill a complete global batch are left unused. Packing may
also discard the usual incomplete sequence tails. Startup logs report
`[TRANCHE] ... unused_tail_samples=...`; the exit logs `[TRANCHE-END]`.

## Resume and move to another tranche

For a crash/time-limit resume of the same tranche, keep the data paths, sequence
length, packing configuration and tranche-start offset unchanged. The checkpoint
consumed-sample counter determines the remaining batches. Replaying a rejected
step does not consume another batch.

When switching to a new tranche, point `DATA_ROOT` at its shards and set:

```bash
--exit-on-tranche-end
--tranche-start-samples <consumed_train_samples_from_boundary_checkpoint>
```

This restarts the tranche-local data cursor at zero while keeping the global
consumed-sample counter, optimizer state and LR schedule. When the explicit offset
is omitted, existing `--phase-transition-iterations` determine the local cursor;
without either setting, the tranche starts at global sample zero.

Enabling this feature requests single-pass dataset indices. Enable it
from the start of a tranche: switching an existing multi-epoch run midway may
change that run's shuffled data order. New index caches may need to be
built; disable `--dataloader-fast-cache-load` for their first build.

Requires `--save`, `--dataloader-type single`, a fixed global batch size, and
GPT pretraining data. Batch rampup, dynamic batch-size decreases, mock data, SFT,
RL and `--rerun-strategy skip_iteration` are rejected.

The launcher `--auto-requeue` is separate from this training exit condition. Do
not use an unconditional `afterany` auto-requeue chain at the final tranche
boundary: it will submit another job for the already-finished tranche. Resume
time-limited jobs manually or make the launcher distinguish `[TRANCHE-END]`
from time-limit exits before scheduling a successor.

## CPU regression checks

```bash
python tests/unit_tests/test_tranche_end.py
```

These checks exercise production boundary, dataset-size, resume, rank broadcast,
and checkpoint-exit logic with distributed/model dependencies stubbed. Validate
the final checkpoint and exit on a short GPU run before a large allocation.
