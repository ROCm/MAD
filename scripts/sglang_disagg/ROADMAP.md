# sglang_disagg config refactor - staged roadmap

This refactor is delivered in three independently reviewable stages. Each stage is
behavior-preserving and builds on the previous one. See [CONFIG.md](CONFIG.md) for the
configuration taxonomy and [DESIGN.md](DESIGN.md) for the high-level design.

## Overview

```mermaid
flowchart LR
    baseline["Baseline<br/>config hard-wired inside scripts"]
    s1["Stage 1<br/>EXTRACT<br/>pull variables out of scripts"]
    s2["Stage 2<br/>CATEGORIZE + VALIDATE<br/>group into the 7 buckets, prove no run regressions"]
    s3["Stage 3<br/>GENERALIZE<br/>folder-per-category, drop-in configs"]

    baseline --> s1 --> s2 --> s3
```

Status: this PR lands Stage 1 + Stage 2 together (the extraction is already bucket-aligned).
Stage 3 is a follow-on this work unblocks.

## Stage 1 - Extract variables out of the scripts

Goal: decouple *values* from *control flow*. Scripts stop hard-coding env/defaults and
instead source external config. Runtime behavior is identical.

```mermaid
flowchart TB
    subgraph before [Before]
        direction TB
        b_slurm["run_xPyD_models.slurm<br/>(-e passthrough + defaults)"]
        b_entry["sglang_disagg_mori_io_ep.sh<br/>(inline VAR:-default)"]
        b_env["mori_ep_env.sh<br/>(all NCCL/MORI env inline)"]
    end

    subgraph after [After Stage 1]
        direction TB
        a_entry["entrypoint (control flow only)"]
        a_env["mori_ep_env.sh (aggregator)"]
        a_cfg["extracted config values<br/>(external, sourced)"]
        a_entry -->|"source"| a_cfg
        a_env -->|"source"| a_cfg
    end

    before ==>|"lift values out"| after
```

Verify: resolved `env` is byte-identical before/after.

## Stage 2 - Categorize into the 7 buckets + validate

Goal: give the extracted values a home aligned to the CONFIG.md taxonomy, and gate on
"no run issues."

```mermaid
flowchart TB
    flat["Extracted config (flat)"]

    subgraph buckets [Bucket-aligned files]
        direction TB
        nic["nic-selection.env.sh<br/>(shared prelude)"]
        fw["framework.env.sh"]
        conn["connectors.env.sh"]
        defs["runtime.defaults.sh<br/>(Cluster/Launcher/Model)"]
    end

    gate{"Validation gate<br/>env-diff across DP_MODE x USE_CX7_NICS<br/>bash -n, standalone sourcing"}

    flat --> nic & fw & conn & defs
    nic --> gate
    fw --> gate
    conn --> gate
    defs --> gate
    gate -->|"identical + clean"| ok["Stage 2 accepted"]
    gate -->|"any diff"| fix["fix + re-verify"]
    fix --> gate
```

## Stage 3 - Generalize into a folder-per-category, drop-in structure

Goal: make it trivial to add a new config - drop a file into the right category folder;
it is auto-discovered and sourced. No script edits needed to add config.

```mermaid
flowchart TB
    subgraph tree [config/ folder-per-bucket]
        direction TB
        cl["cluster/*.env.sh"]
        fw2["framework/*.env.sh"]
        cn["connectors/*.env.sh"]
        md["model/*.env.sh"]
        bn["benchmarking/*.env.sh"]
    end

    loader["loader: source category/*.env.sh in dependency order"]
    newcfg["new requirement"]

    cl --> loader
    fw2 --> loader
    cn --> loader
    md --> loader
    bn --> loader
    loader --> entry["entrypoint / aggregator"]

    newcfg -.->|"add one file, no code change"| fw2
```

Extensibility contract: adding a backend / model / tuning knob = add a single file under
the matching category folder; the loader picks it up automatically.
