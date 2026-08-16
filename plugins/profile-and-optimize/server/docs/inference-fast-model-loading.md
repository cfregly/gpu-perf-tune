Status: Active
Audience: operators loading large model checkpoints for inference.

# Fast model loading for inference

Large checkpoints should not load through a single slow FUSE stream by default. Measure the available paths, choose the fastest verified option, and record the result with the serving configuration.

## Preferred order

1. Use a model streamer supported by the serving runtime and object store.
2. Stage shards in parallel to local NVMe, then load from the local path.
3. Use a pre-serialized format when the artifact and runtime support it.
4. Use a direct remote filesystem only after measuring it against the other options.

Every choice needs an availability check and a measured effective transfer rate. A hostname resolving or a TCP port opening does not prove that the endpoint can read the target bucket.

## Parallel staging

[`stage-model-parallel.py`](../tools/stage-model-parallel.py) downloads objects concurrently with multipart transfers. [`local-nvme-probe.sh`](../tools/local-nvme-probe.sh) checks local capacity before staging.

```bash
python3 plugins/profile-and-optimize/server/tools/stage-model-parallel.py \
  s3://example-bucket/model-prefix /models/example-model
```

Provide object-store credentials through the environment or the workload's secret manager. Do not write credentials into commands, evidence bundles, or repository files.

## Loader selection

[`loader_advisor.py`](../tools/loader_advisor.py) selects a loader from explicit facts about the serving tier:

- whether the checkpoint includes an in-checkpoint speculative decoder
- whether the image supports a model streamer
- whether the object store and model artifact are reachable
- whether local storage has enough capacity
- whether the deployment allows external model-registry access

```bash
python3 plugins/profile-and-optimize/server/tools/loader_advisor.py \
  --serve-args "<serving arguments>" \
  --hf-egress yes \
  --image-has-runai yes \
  --s3 yes
```

## Validation

Record checkpoint size, wall-clock load time, effective throughput, storage path, image digest, and serving arguments. Re-run the measurement after changing the image, storage endpoint, checkpoint layout, or loader. Treat example values as local observations, not portable performance claims.
