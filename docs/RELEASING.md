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
#    Set release_version!() in crates/anofox-bayes-core/src/lib.rs and the
#    ExtensionVersion()/banner literals in src/anofox_bayes_extension.cpp to
#    today's date, then move the CHANGELOG Unreleased section under it.
./scripts/check_version.sh                             # all three agree

# 3. Tag and push. CalVer: v$(date +%Y.%m.%d).
git tag -a v2026.08.10 -m "anofox-bayes v2026.08.10"
git push origin main --follow-tags
```

## Versioning: CalVer, and why not Cargo.toml

Releases are dated — `vYYYY.MM.DD` — matching `erpl`, `erpl-idoc`, `erpl-tunnel`,
`gdrive` and the rest of the fleet. A dated version makes no compatibility claim, so
anything that would have been a breaking change under semver has to be spelled out in
the CHANGELOG. The **draws schema** is where compatibility is actually promised, and it
stays a separate integer (`docs/DRAWS_CONTRACT.md`).

**The number cannot live in `Cargo.toml`.** Cargo parses the manifest version as semver
and rejects a zero-padded date:

```console
$ cargo metadata          # with version = "2026.08.10"
error: invalid leading zero in minor version number
```

Only `2026.8.10` parses, and that is a different string from the tag — so the manifest
keeps a semver number that is crate metadata and **deliberately not** the release
identity. The release version is a macro in `crates/anofox-bayes-core/src/lib.rs`:

```rust
macro_rules! release_version { () => { "2026.08.10" }; }
pub const VERSION: &str   = release_version!();
pub const VERSION_C: &str = concat!(release_version!(), "\0");   // for the FFI
```

A macro rather than a `const` because the FFI needs a *literal* to build the
NUL-terminated string at compile time — `concat!` takes literals, not constants — and
one source of truth beats two that can drift.

`anofox_bayes_version()` still reads through the FFI, so it remains the end-to-end
proof that the Rust core is linked in; it now answers with the release date.

**Three copies must agree** — the Rust macro, the C++ `ExtensionVersion()` fallback and
the banner literal — plus the tag on a tag build. `scripts/check_version.sh` checks all
of them and runs in CI. Without it a stale constant is invisible: the extension builds,
loads, and reports the *previous* release while the S3 path carries the new tag.

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
aws s3 ls s3://get.erpl.io/anofox_bayes/v2026.08.10/ --recursive | head

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
