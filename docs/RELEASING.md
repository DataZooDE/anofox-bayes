# Releasing and self-distribution

`anofox-bayes` is BSL-licensed, so it cannot go to the DuckDB community repository.
It publishes to DataZoo's own channel, the same one `erpl`, `anofox-statistics` and
`anofox-forecast` use.

```sql
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';
LOAD anofox_bayes;
```

## How the channel works

| Piece | Value |
|---|---|
| S3 bucket | `get.erpl.io` (repository **variable** `DEPLOY_S3_BUCKET`) |
| AWS account | `331993160594`, region `eu-west-1` |
| Auth | GitHub OIDC → `arn:aws:iam::331993160594:role/ErplGithubOicdRole` |
| Upload script | `scripts/extension-upload.sh` |
| Layout | `s3://get.erpl.io/<ext>/<ext_version>/<duckdb_version>/<arch>/<ext>.duckdb_extension.gz` |

No long-lived AWS keys exist anywhere: the workflow exchanges a GitHub OIDC token for
temporary credentials. The role's trust policy is organisation-wide, so a new
DataZooDE repository needs no IAM change — only the bucket variable.

## Setting up a new repository

**One thing**, in this organisation.

### 1. The bucket variable

```bash
gh variable set DEPLOY_S3_BUCKET --body "get.erpl.io" --repo DataZooDE/<repo>
```

A repository *variable*, not a secret — it is a public bucket name.

### 2. The IAM trust policy — usually nothing to do

`ErplGithubOicdRole`'s trust policy is **organisation-wide**:

```json
"StringLike": {
  "token.actions.githubusercontent.com:sub": [
    "repo:DataZooDE/*",
    "repo:DataZooDE@136052936/*"
  ]
}
```

IAM `StringLike` wildcards match across `:` and `/`, so `repo:DataZooDE/*` already
covers `repo:DataZooDE/<any-repo>:ref:refs/heads/main`. **A new repository under the
DataZooDE organisation needs no IAM change at all.**

This is worth stating plainly because `AccessDenied` on every deploy job *looks* like
a permissions problem and is much more often an unset `DEPLOY_S3_BUCKET` — see the
failure table below. Only a repository outside the organisation would need a new
entry, which would look like:

```json
{
  "Effect": "Allow",
  "Principal": {
    "Federated": "arn:aws:iam::331993160594:oidc-provider/token.actions.githubusercontent.com"
  },
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {
    "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
    "StringLike": {
      "token.actions.githubusercontent.com:sub": [
        "repo:DataZooDE/erpl:*",
        "repo:DataZooDE/anofox-statistics:*",
        "repo:DataZooDE/anofox-forecast:*",
        "repo:DataZooDE/anofox-bayes:*"
      ]
    }
  }
}
```

Read the current policy before editing — it is shared, and the list above is
illustrative rather than authoritative:

```bash
aws iam get-role --role-name ErplGithubOicdRole \
  --query 'Role.AssumeRolePolicyDocument'
```

`MainDistributionPipeline.yml` skips the deploy jobs entirely while
`DEPLOY_S3_BUCKET` is unset, so a fresh fork is quiet rather than red.

## Cutting a release

Deployment triggers on a push to `main` (publishing as *latest*) or on a `v*` tag
(publishing a pinned version as well).

```bash
# 1. Everything green locally first.
cargo test --workspace && make lint && make test
cargo test --workspace --release -- --ignored          # calibration suites
(cd validation && uv run pytest -q)                    # PyMC parity

# 2. Version and changelog.
#    Bump `version` in Cargo.toml [workspace.package]; move the CHANGELOG
#    Unreleased section under the new number.

# 3. Tag and push.
git tag -a v0.1.0 -m "anofox-bayes v0.1.0"
git push origin main --follow-tags
```

**Use annotated tags** (`-a`). Lightweight ones work too, but the deploy job's
`git fetch --tags --force` exists precisely because an annotated tag once failed the
whole pipeline — `actions/checkout` leaves `refs/tags/<t>` pointing at the commit, and
fetching the annotated tag then wants to replace it with a tag object, which git
refuses as "would clobber existing tag".

## Status

First publication: **2026-08-01**, from commit `9a91d22` on `main`. Verified by
installing on a stock DuckDB v1.5.5 CLI with a clean home directory. Artifacts exist
for both DuckDB versions on linux/macOS/Windows (amd64 + arm64) and WASM.

## Verifying a release

```bash
# Binaries landed for the architectures you expect:
aws s3 ls s3://get.erpl.io/anofox_bayes/v0.1.0/ --recursive | head

# And it installs from a clean DuckDB:
duckdb -c "INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';
           LOAD anofox_bayes;
           SELECT anofox_bayes_version();"
```

## Things that have gone wrong before

Recorded so the next repository does not rediscover them.

| Symptom | Cause |
|---|---|
| `AccessDenied` on every deploy job, builds all green | **Almost always `DEPLOY_S3_BUCKET` is unset**, not permissions. Unquoted, the empty argument vanishes and the rest shift left, so the upload goes to a bucket named `true`. Look for `s3://true/` in the log. Both the workflow (quoting) and `extension-upload.sh` (an argument guard) now refuse this rather than trying it |
| `would clobber existing tag`, deploy fails before uploading | Missing `git fetch --tags --force` |
| `metadata at the end of the file is invalid` on install | The 256-byte signature placeholder was not stripped before re-appending. `scripts/extension-upload.sh` handles this; `erpl-tunnel`'s older copy does not |
| Deploy runs but publishes nothing | `deploy_latest` / `deploy_versioned` are false off `main` and off a `v*` tag — expected on a branch |
| A missing artifact cancels sibling architectures | `fail-fast: false` on the deploy matrix; already set here |

Binaries are **unsigned** — `DUCKDB_EXTENSION_SIGNING_PK` is not configured, so
installs need `allow_unsigned_extensions` unless DuckDB is started with the
community-extension trust settings.
