#!/usr/bin/env python3
"""Parallel multipart model staging: S3-compatible object storage -> local NVMe.

The inference-side workhorse for the "never settle for slow loads" rule
(`docs/METHODOLOGY.md`). Replaces single-stream s3fs FUSE
(GLM-5.1 NVFP4 = 433.9 GB @ ~58 MB/s = ~50 min) with parallel boto3 multipart
downloads (N concurrent objects x M-way multipart); vLLM then loads from local
NVMe. Validated 2026-06-08 at ~10x the s3fs rate on a GB300 node.

Endpoint selection (preference order: in-cluster accelerated endpoint -> global):
  - If S3_ENDPOINT_URL is set, it is used verbatim.
  - Else auto-probe the in-cluster accelerated endpoint (ACCEL_S3_ENDPOINT_URL); if reachable, use it (faster);
    otherwise fall back to the global S3-compatible endpoint (S3_FALLBACK_URL).
Some providers require virtual-hosted-style addressing (bucket.host/key); path-style
is rejected on ListObjectsV2 (PathStyleRequestNotAllowed).

Env (perflake-s3-creds supplies the creds):
  PERFLAKE_LAKE_S3_ACCESS_KEY / PERFLAKE_LAKE_S3_SECRET_KEY  (or AWS_ACCESS_KEY_ID/SECRET)
  S3_ENDPOINT_URL    explicit endpoint (skips the accelerated-endpoint auto-probe)
  ACCEL_S3_ENDPOINT_URL     default http://accelerated-object-store.example.com (set to your provider's in-cluster endpoint)
  S3_FALLBACK_URL    default https://object-store.example.com (set to your provider's S3-compatible endpoint)
  S3_REGION          default <zone>
  STAGE_SHARD_CONCURRENCY      default 16   (concurrent objects)
  STAGE_MULTIPART_CONCURRENCY  default 8    (streams per object)
  STAGE_CHUNK_MB               default 256  (multipart chunk size)

Usage:
  stage-model-parallel.py s3://perf-lake/saved_experiments/cluster-pvcs/glm51-cache/target /models/glm51
"""
import os
import sys
import time
import socket
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

# Fast-model-loading endpoint candidates. Some GPU-cloud providers expose
# an in-cluster accelerated object-storage endpoint that only resolves
# inside the cluster; when it is unreachable the probe falls back to the
# provider's global S3-compatible endpoint. Both are env-configurable —
# any S3-compatible endpoint works (set ACCEL_S3_ENDPOINT_URL /
# S3_FALLBACK_URL to your provider's hosts).
ACCEL_S3_CANDIDATES = [
    os.environ.get("ACCEL_S3_ENDPOINT_URL", "http://accelerated-object-store.example.com"),
]
FALLBACK_URL = os.environ.get("S3_FALLBACK_URL", "https://object-store.example.com")
REGION = os.environ.get("S3_REGION", "<zone>")
AK = os.environ.get("PERFLAKE_LAKE_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
SK = os.environ.get("PERFLAKE_LAKE_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
SHARD_CONCURRENCY = int(os.environ.get("STAGE_SHARD_CONCURRENCY", "16"))
MULTIPART_CONCURRENCY = int(os.environ.get("STAGE_MULTIPART_CONCURRENCY", "8"))
CHUNK_MB = int(os.environ.get("STAGE_CHUNK_MB", "256"))


def _reachable(url: str, timeout: float = 5.0) -> bool:
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def pick_endpoint() -> str:
    explicit = os.environ.get("S3_ENDPOINT_URL")
    if explicit:
        print(f"[stage] endpoint (explicit): {explicit}", flush=True)
        return explicit
    for url in ACCEL_S3_CANDIDATES:
        if _reachable(url):
            print(f"[stage] endpoint: accelerated {url} (reachable)", flush=True)
            return url
    print(f"[stage] endpoint: global {FALLBACK_URL} (no accelerated candidate reachable)", flush=True)
    return FALLBACK_URL


def parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise SystemExit(f"FATAL: source must be s3://bucket/prefix, got {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.rstrip("/")


def make_client(endpoint: str):
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=AK,
        aws_secret_access_key=SK, region_name=REGION,
        config=Config(
            s3={"addressing_style": "virtual"},  # some providers reject path-style on ListObjectsV2
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=MULTIPART_CONCURRENCY * 2 + 8,
        ),
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: stage-model-parallel.py s3://bucket/prefix /local/dir")
    src, dst = sys.argv[1], sys.argv[2]
    bucket, prefix = parse_s3(src)
    if not AK or not SK:
        raise SystemExit("FATAL: missing S3 creds (PERFLAKE_LAKE_S3_ACCESS_KEY/SECRET_KEY)")
    os.makedirs(dst, exist_ok=True)
    endpoint = pick_endpoint()

    cfg = TransferConfig(
        max_concurrency=MULTIPART_CONCURRENCY,
        multipart_threshold=CHUNK_MB * 1024 * 1024,
        multipart_chunksize=CHUNK_MB * 1024 * 1024,
        use_threads=True,
    )

    s3 = make_client(endpoint)
    keys = []
    total = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            keys.append((o["Key"], o["Size"]))
            total += o["Size"]
    if not keys:
        raise SystemExit(f"FATAL: no objects under s3://{bucket}/{prefix}/")

    print(f"[stage] {len(keys)} objects, {total/1e9:.1f} GB  ->  {dst}", flush=True)
    print(f"[stage] shards={SHARD_CONCURRENCY} multipart={MULTIPART_CONCURRENCY}x{CHUNK_MB}MB", flush=True)

    t0 = time.time()
    done = {"n": 0, "bytes": 0}
    lock = threading.Lock()
    tl = threading.local()

    def fetch(key, size):
        rel = key[len(prefix) + 1:] if key.startswith(prefix + "/") else os.path.basename(key)
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out) or dst, exist_ok=True)
        if not hasattr(tl, "c"):
            tl.c = make_client(endpoint)
        tl.c.download_file(bucket, key, out, Config=cfg)
        got = os.path.getsize(out)
        if got != size:
            raise RuntimeError(f"size mismatch {rel}: got {got} expected {size}")
        with lock:
            done["n"] += 1
            done["bytes"] += size
            el = max(time.time() - t0, 1e-6)
            print(f"[stage] {done['n']}/{len(keys)} {rel} ({size/1e9:.2f}GB) | "
                  f"{done['bytes']/1e9:.1f}/{total/1e9:.1f}GB  {done['bytes']/1e6/el:.0f} MB/s agg", flush=True)

    with ThreadPoolExecutor(max_workers=SHARD_CONCURRENCY) as ex:
        futs = [ex.submit(fetch, k, s) for k, s in keys]
        for f in as_completed(futs):
            f.result()  # re-raise any worker error (fail-loud, no partial model)

    el = max(time.time() - t0, 1e-6)
    print(f"[stage] DONE {total/1e9:.1f} GB in {el:.0f}s = {total/1e6/el:.0f} MB/s aggregate  ->  {dst}", flush=True)


if __name__ == "__main__":
    main()
