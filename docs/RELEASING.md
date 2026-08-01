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

No long-lived AWS keys exist anywhere. The workflow exchanges a GitHub OIDC token for
temporary credentials, which is why **the role's trust policy has to name each
repository explicitly**.

## Setting up a new repository

Two things, and the second is the one that is easy to forget.

### 1. The bucket variable

```bash
gh variable set DEPLOY_S3_BUCKET --body "get.erpl.io" --repo DataZooDE/<repo>
```

A repository *variable*, not a secret — it is a public bucket name.

### 2. The IAM trust policy  ← needs AWS console/CLI access

`ErplGithubOicdRole` will refuse the token until its trust policy allows the new
repository. Until then every deploy job fails `AccessDenied` *after* a full build
matrix has run, which is an expensive way to discover it.

Add the repository to the `sub` condition:

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

**Until this is done**, `MainDistributionPipeline.yml` skips the deploy jobs rather
than failing them, provided `DEPLOY_S3_BUCKET` is unset. Once the variable is set the
gate opens, so set the variable and the trust policy together.

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
| `AccessDenied` on every deploy job, builds all green | The trust policy does not name this repository |
| `would clobber existing tag`, deploy fails before uploading | Missing `git fetch --tags --force` |
| `metadata at the end of the file is invalid` on install | The 256-byte signature placeholder was not stripped before re-appending. `scripts/extension-upload.sh` handles this; `erpl-tunnel`'s older copy does not |
| Deploy runs but publishes nothing | `deploy_latest` / `deploy_versioned` are false off `main` and off a `v*` tag — expected on a branch |
| A missing artifact cancels sibling architectures | `fail-fast: false` on the deploy matrix; already set here |

Binaries are **unsigned** — `DUCKDB_EXTENSION_SIGNING_PK` is not configured, so
installs need `allow_unsigned_extensions` unless DuckDB is started with the
community-extension trust settings.
