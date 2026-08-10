#!/usr/bin/env bash
# The release version is written in three places that must agree, and nothing
# else notices when they do not:
#
#   crates/anofox-bayes-core/src/lib.rs  release_version!()   -> the SQL surface
#   src/anofox_bayes_extension.cpp       ExtensionVersion()   -> local-build fallback
#   the git tag                          vYYYY.MM.DD          -> the published path
#
# A stale constant is invisible: the extension builds, loads, and answers
# `SELECT anofox_bayes_version()` with the *previous* release, while the S3 path
# carries the new tag. On a tag build this script also checks the tag itself, so
# cutting a release with a forgotten bump fails the pipeline instead of shipping.
#
# CalVer, YYYY.MM.DD. The version cannot live in Cargo.toml: cargo parses that
# as semver and rejects `2026.08.10` -- invalid leading zero in the minor number.
set -euo pipefail

cd "$(dirname "$0")/.."

rust=$(grep -oE '"[0-9]{4}\.[0-9]{2}\.[0-9]{2}"' crates/anofox-bayes-core/src/lib.rs | head -1 | tr -d '"')
cpp=$(grep -oE 'return "[0-9]{4}\.[0-9]{2}\.[0-9]{2}"' src/anofox_bayes_extension.cpp | head -1 | grep -oE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
banner=$(grep -oE '"anofox_bayes", "[0-9]{4}\.[0-9]{2}\.[0-9]{2}"' src/anofox_bayes_extension.cpp | grep -oE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}')

fail() { echo "FAIL: $1" >&2; exit 1; }

[[ -n "$rust"   ]] || fail "no CalVer version found in crates/anofox-bayes-core/src/lib.rs"
[[ -n "$cpp"    ]] || fail "no CalVer fallback found in src/anofox_bayes_extension.cpp"
[[ -n "$banner" ]] || fail "no CalVer version found in the banner definition"

echo "rust   : $rust"
echo "cpp    : $cpp"
echo "banner : $banner"

[[ "$rust" == "$cpp" ]]    || fail "rust ($rust) != cpp fallback ($cpp)"
[[ "$rust" == "$banner" ]] || fail "rust ($rust) != banner ($banner)"

# On a tag build, the tag is the fourth place and the one users see.
tag="${GITHUB_REF_NAME:-}"
if [[ "$tag" == v* ]]; then
	echo "tag    : $tag"
	[[ "v$rust" == "$tag" ]] || fail "tag $tag does not match the source version v$rust"
fi

echo "OK: version is $rust everywhere"
