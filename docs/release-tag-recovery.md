# Release tag recovery procedure

When the `vX.Y.Z` tag you plan to ship has already been claimed by a parallel
unrelated commit on `origin/main`, here is the recovery procedure.

## How this happens

Two workstreams target the same next version. While one is still in flight,
an unrelated PR merges first, bumps `plugin.json` to that version, and tags
the merge commit. By the time the in-flight work is ready to commit, the
tag is already taken and `main` has diverged.

The recovery: cherry-pick the in-flight work onto `origin/main`, resolve the
version / banner conflicts, and ship as the next PATCH.

## Recovery procedure (general form)

If your local `main` is both ahead of and behind `origin/main` and the version
tag you intended to ship already exists on `origin`:

### 1. Confirm the state (read-only)

```bash
git fetch origin
git status -sb                            # ahead/behind counters
git log HEAD..origin/main --oneline       # commits I don't have locally
git log origin/main..HEAD --oneline       # commits I have but origin doesn't
git ls-remote --tags origin vX.Y.Z        # is the tag on the remote?
git show vX.Y.Z --stat --no-patch         # what's at the existing tag?
```

If the existing `vX.Y.Z` tag is on the remote AND was authored by someone
else, **DO NOT force-push the tag**. Bump to `vX.Y.(Z+1)` PATCH or
`vX.(Y+1).0` MINOR instead.

### 2. Reset to origin/main + cherry-pick your commits

```bash
# Your local commits are preserved in the reflog (git reflog | head),
# so reset --hard is safe.
git reset --hard origin/main

# Cherry-pick each in-flight commit. If both commits modify overlapping
# files (typically plugin.json + README banners), the cherry-pick will
# conflict; resolve manually (see step 3).
git cherry-pick <commit-1-sha>
# resolve conflicts ...
git cherry-pick --continue
git cherry-pick <commit-2-sha>
# resolve conflicts ...
git cherry-pick --continue
```

### 3. Resolve the typical version / banner conflicts

Three conflict patterns recur:

1. **Release-notes section header**: Both your cherry-pick and the existing
   tag claim the same version. Resolution: **rename your section to
   `X.Y.(Z+1)`**, keep the existing `X.Y.Z` notes unchanged.
2. **README.md "Status:" banner**: Both versions update the same banner line
   with different content. Resolution: **merge both summaries into one
   banner**, mentioning the new version's headline first and the previous
   version's contribution second ("vX.Y.(Z+1) ships A + B. VX.Y.Z shipped C").
3. **plugin.json `"version"`**: The merged commit set it to the old version
   (X.Y.Z). Your cherry-pick was prepared for the same version. Resolution:
   bump to the new patch version (X.Y.Z+1).

### 4. Re-run gates BEFORE tagging

```bash
make all
# pytest, check-doc-links, lint-skill-mcp-args, mcp-surface
# all must be GREEN; if not, fix conflicts left over from the merge
```

A common late-surfacing failure is `lint-skill-mcp-args` going RED because
the merge composed two unrelated changes (e.g., one removes servers from
`.mcp.json`, another updates the lint script to detect optional servers
*from* `.mcp.json`). Fix the composed bug as part of the recovery commit,
do NOT defer.

### 5. Tag the new version + push

```bash
git tag -a vX.Y.(Z+1) -m "profile-and-optimize vX.Y.(Z+1): <summary>"
git push origin main
git push origin vX.Y.(Z+1)
gh release create vX.Y.(Z+1) --title "..." --notes "$(cat <<'EOF'
...
EOF
)"
```

## Prevention: pre-tag checks

Before bumping `plugin.json` + tagging, always run:

```bash
git fetch origin
git status -sb              # if ahead != 0 OR behind != 0, STOP
git ls-remote --tags origin <target-tag>   # if non-empty, the tag is already taken
```

If `git status -sb` shows `behind > 0` OR the tag already exists on origin,
do NOT proceed with the tag bump. Either rebase + bump the patch version, or
coordinate with the other workstream owner on how to merge.

## What a clean recovery preserves

- All gates GREEN (pytest, check-doc-links, lint-skill-mcp-args, mcp-surface
  unchanged).
- Backward compat: any deprecation aliases in the cherry-picked work survive
  the recovery (keep legacy env vars as deprecation warnings, not hard
  removals).
- Evidence bundles ship intact under their original names. Only the release
  tag name shifts.

## Cross-references

- [`CONTRIBUTING.md`](/CONTRIBUTING.md#release-ritual) "Release ritual" -- standard release flow that this recovery wraps when the pre-tag check fails.
