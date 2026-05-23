#!/usr/bin/env bash
set -euo pipefail

# bump_and_publish.sh — bump the caps-sa version in Cargo.toml, commit + tag
# + push, and optionally `cargo publish` to crates.io.
#
# caps-sa is a library crate; Cargo.lock is gitignored (see .gitignore), so
# this script only edits Cargo.toml. The workspace-level lockfile sitting one
# directory up is regenerated automatically by `cargo check` / `cargo build`
# runs in the parent workspace and is not part of the crate's git history.

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  ./bump_and_publish.sh <version> [--publish] [--dry-run]
  ./bump_and_publish.sh [--publish] [--dry-run] <version>

Options:
  --publish  Publish to crates.io after bumping, committing, tagging, and pushing
  --dry-run  Show what would be done without modifying Cargo.toml, creating
             commits or tags, pushing, or publishing
  -h, --help Show this help message
EOF
}

print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_cmd "$@"
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    "$@"
}

VERSION=""
PUBLISH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --publish)
            PUBLISH=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -n "$VERSION" ]]; then
                die "version specified more than once"
            fi
            VERSION="$1"
            ;;
    esac
    shift
done

[[ -n "$VERSION" ]] || {
    usage
    exit 1
}

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)*$ ]]; then
    die "version must look like X.Y.Z, optionally with prerelease/build suffixes"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT_CARGO="Cargo.toml"
TAG="v${VERSION}"
CRATE_NAME="caps-sa"
TMP_TARGET_DIR=""
MANIFEST_BACKUP=""
MANIFEST_UPDATED=false
COMMIT_CREATED=false

cleanup() {
    local status=$?

    if [[ -n "$TMP_TARGET_DIR" && -d "$TMP_TARGET_DIR" ]]; then
        rm -rf "$TMP_TARGET_DIR"
    fi

    # If anything failed after we rewrote Cargo.toml but before the release
    # commit landed, restore the manifest from backup so the working tree
    # is left as we found it.
    if [[ "$status" -ne 0 && "$DRY_RUN" == false && "$MANIFEST_UPDATED" == true && "$COMMIT_CREATED" == false ]]; then
        if [[ -n "$MANIFEST_BACKUP" && -f "$MANIFEST_BACKUP" ]]; then
            cp "$MANIFEST_BACKUP" "$ROOT_CARGO"
            echo "restored $ROOT_CARGO after failure" >&2
        fi
    fi

    if [[ -n "$MANIFEST_BACKUP" && -f "$MANIFEST_BACKUP" ]]; then
        rm -f "$MANIFEST_BACKUP"
    fi

    return "$status"
}

trap cleanup EXIT

[[ -f "$ROOT_CARGO" ]] || die "not found: $ROOT_CARGO"

CURRENT_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT_CARGO" | head -1)"
[[ -n "$CURRENT_VERSION" ]] || die "could not determine current crate version from $ROOT_CARGO"

if [[ "$CURRENT_VERSION" == "$VERSION" ]]; then
    die "crate version is already $VERSION"
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists"
fi

if [[ -n "$(git status --porcelain)" ]]; then
    die "working tree is not clean; commit or stash existing changes first"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    die "git remote 'origin' is not configured"
fi

echo "Current crate version : $CURRENT_VERSION"
echo "New crate version     : $VERSION"
echo "Tag                   : $TAG"
if [[ "$PUBLISH" == true ]]; then
    echo "Publish               : yes"
else
    echo "Publish               : no"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run               : yes"
else
    echo "Dry-run               : no"
fi
echo

# Preflight: make sure the unbumped crate builds + packages cleanly. Doing
# this first means we catch dependency / manifest issues before touching the
# version string.
echo "Preflight checks before changing version"
cargo check -q
TMP_TARGET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/caps-sa-release-check.XXXXXX")"
CARGO_TARGET_DIR="$TMP_TARGET_DIR" cargo package --offline --allow-dirty --no-verify >/dev/null
rm -rf "$TMP_TARGET_DIR"
TMP_TARGET_DIR=""

echo "Updating $ROOT_CARGO"
echo "  version: $CURRENT_VERSION -> $VERSION"

if [[ "$DRY_RUN" == false ]]; then
    MANIFEST_BACKUP="$(mktemp "${TMPDIR:-/tmp}/caps-sa-Cargo.toml.XXXXXX")"
    cp "$ROOT_CARGO" "$MANIFEST_BACKUP"

    # Rewrite only the first `version = "..."` line — the one inside the
    # [package] table at the top of the manifest. Workspace dependency
    # versions (if any are pinned with `version = "..."` syntax) appear
    # later and are intentionally left alone.
    sed -i.bak "1,/^version = /s/^version = \".*\"/version = \"${VERSION}\"/" "$ROOT_CARGO"
    rm -f "${ROOT_CARGO}.bak"

    MANIFEST_UPDATED=true
else
    echo "Dry-run: would rewrite $ROOT_CARGO"
fi

if [[ "$DRY_RUN" == false ]]; then
    UPDATED_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT_CARGO" | head -1)"
    [[ "$UPDATED_VERSION" == "$VERSION" ]] || die "crate version update failed"
fi

echo
echo "Post-bump validation"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run: would run cargo check and cargo package against the bumped version"
else
    cargo check -q
    TMP_TARGET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/caps-sa-release-check.XXXXXX")"
    CARGO_TARGET_DIR="$TMP_TARGET_DIR" cargo package --offline --allow-dirty --no-verify >/dev/null
    rm -rf "$TMP_TARGET_DIR"
    TMP_TARGET_DIR=""
fi

run git add "$ROOT_CARGO"
run git commit -m "chore(release): bump ${CRATE_NAME} to v${VERSION}"

if [[ "$DRY_RUN" == false ]]; then
    COMMIT_CREATED=true
fi

run git tag -a "$TAG" -m "Release ${VERSION}"
run git push origin HEAD
run git push origin "$TAG"

if [[ "$PUBLISH" == true ]]; then
    run cargo publish
else
    echo "Skipping crates.io publish; re-run with --publish to publish ${CRATE_NAME} v${VERSION}"
fi

echo
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run complete"
else
    echo "Release bump and publish complete for v${VERSION}"
fi
