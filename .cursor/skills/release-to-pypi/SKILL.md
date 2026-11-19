---
name: release-to-pypi
description: Release the Solarian core or a Solarian plugin to PyPI by bumping its version and creating a GitHub release, which triggers the repo's publish workflow. Also onboards a plugin that has never been published. Use when the user asks to release, publish, ship, cut a version of, or push to PyPI solarian or any solarian-* plugin.
---

# Release to PyPI

Each package is its own GitHub repo under `solarian-dev`. Publishing a GitHub
release `vX.Y.Z` runs `.github/workflows/publish.yml`, which checks that the tag
matches `[project].version` in `pyproject.toml`, builds the sdist and wheel, and
uploads them with PyPI Trusted Publishing (no tokens). The upload job runs in the
repo's `pypi` environment, which only accepts `v*` tags.

| Package | Local path | Repo |
| --- | --- | --- |
| `solarian` | `.` (this repo) | `solarian-dev/solarian` |
| `solarian-<name>` | `plugins/solarian-<name>` | `solarian-dev/solarian-<name>` |

A package is set up for publishing if it has `.github/workflows/publish.yml`.
If it does not, follow "Onboarding a new package" first.

## Release workflow

```
- [ ] 1. Preflight
- [ ] 2. Pick the version
- [ ] 3. Check dependencies are released
- [ ] 4. Bump, sync the agent plugin (core), build, commit, push
- [ ] 5. Create the release
- [ ] 6. Verify
```

**1. Preflight** (in the package's repo)
- `git status -sb`: on `main` with a clean tree. If the user has uncommitted
  changes, ask before committing them.
- `git fetch && git status -sb`: not behind `origin/main`.
- Current published version: `curl -s https://pypi.org/simple/<package>/` and
  read the newest filename.
- Core only: `python integrations/sync_agent_plugin.py --check`. If it lists
  files, someone changed `solarian/templates/agent/bootstrap/` without syncing;
  run the sync now and commit it before the release commit.

**2. Pick the version**
- Ask the user for the bump (patch / minor / major) unless they named one.
  Pre-1.0: breaking changes are a minor bump.
- PyPI never accepts a version twice, even after deleting it. The new version
  must be higher than every version listed in the simple index.

**3. Check dependencies are released**
- If the change relies on unreleased code in `solarian` or another plugin
  (e.g. `solarian-feetech`), release that package first and raise the lower
  bound in `dependencies` (e.g. `"solarian>=0.3.0"`).
- `solarian-robstride` is not on PyPI; it stays an optional extra.

**4. Bump, build, commit, push**
- Set `version` in `pyproject.toml`, and `__version__` in the package's
  `__init__.py` if it defines one.
- Core only: run the agent plugin sync after the bump, every release:

  ```bash
  python integrations/sync_agent_plugin.py
  ```

  It writes the new version into `integrations/agent-plugin/.cursor-plugin/plugin.json`
  and `.claude-plugin/plugin.json` and copies the bootstrap skill into
  `integrations/agent-plugin/skills/`. Those files go in the release commit:
  the Cursor and Claude Code marketplaces read them from `main`, and
  `tests/test_agent_files.py::test_the_marketplace_plugin_matches_the_templates`
  fails until they match. Plugins have nothing to sync; their `agent/` files ship
  in the wheel.
- Build locally as a check. Core uses a custom backend (`solarian_build.py`)
  that compiles atomics on Apple Silicon, so build only the sdist there:

  ```bash
  python -m build --sdist --outdir /tmp/release-check .      # core
  python -m build --outdir /tmp/release-check <plugin path>  # plugin
  python -m twine check /tmp/release-check/*
  ```

  For a plugin with assets, list the wheel (`unzip -l`) and confirm the files
  in `[tool.setuptools.package-data]` are present, including
  `<package>/agent/AGENTS.md` and `agent/skills/*/SKILL.md` when the plugin
  ships them. Without them `solarian agent sync` finds nothing for the plugin.
- Commit as `Release vX.Y.Z` and push to `origin main`.

**5. Create the release**

```bash
gh release create vX.Y.Z -R solarian-dev/<repo> --target main --title "vX.Y.Z" --generate-notes
```

**6. Verify**

```bash
id=$(gh run list -R solarian-dev/<repo> -L 1 --json databaseId -q '.[0].databaseId')
gh run watch $id -R solarian-dev/<repo> --exit-status --interval 10
curl -s https://pypi.org/simple/<package>/ | grep -o '<dist_name>-X.Y.Z[^"#<]*' | sort -u
```

Check the simple index, not `https://pypi.org/pypi/<package>/json`: the JSON
API caches a 404 for several minutes if it was queried before the upload.

For a larger release, smoke-test in a fresh venv:
`pip install --no-cache-dir <package>==X.Y.Z` and import the main module.

Report the PyPI URL (`https://pypi.org/project/<package>/X.Y.Z/`) and the run URL.

## When the workflow fails

- **Tag/version mismatch**: delete the release and tag
  (`gh release delete vX.Y.Z -R solarian-dev/<repo> --cleanup-tag --yes`),
  fix `pyproject.toml`, push, release again.
- **Build failure**: same as above after fixing the build.
- **`invalid-publisher` at upload**: the PyPI trusted publisher is missing or
  its fields differ from the workflow. The user must fix it on PyPI; then
  `gh run rerun <id> -R solarian-dev/<repo> --failed`.
- **Upload succeeded for one file and failed for the other**: the version is
  burnt. Bump the patch version and release again.

## Onboarding a new package

1. Copy the workflow from an onboarded package:
   `cp plugins/solarian-feetech/.github/workflows/publish.yml plugins/solarian-<name>/.github/workflows/`
2. Check the `pyproject.toml` is publishable: `license`, `readme`, `authors`,
   `LICENSE` and `NOTICE` files present, assets listed in
   `[tool.setuptools.package-data]` (with `"agent/*.md", "agent/skills/*/*"` if
   the plugin has an `agent/` folder), and every dependency available on PyPI
   (move unpublished ones such as `solarian-robstride` into an extra).
   README links to repo files should be absolute GitHub URLs; relative ones
   break on the PyPI page.
3. Create the GitHub environment, restricted to `v*` tags:

   ```bash
   gh api -X PUT repos/solarian-dev/<repo>/environments/pypi --input - <<<'{"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}'
   gh api -X POST repos/solarian-dev/<repo>/environments/pypi/deployment-branch-policies -f name='v*' -f type=tag
   ```

4. Ask the user to add the trusted publisher on PyPI; only they can. Fields:
   owner `solarian-dev`, repo `<repo>`, workflow `publish.yml`, environment `pypi`.
   - Name not on PyPI yet: "Add a new pending publisher" at
     https://pypi.org/manage/account/publishing/ — no manual first upload needed.
   - Project already exists: https://pypi.org/manage/project/<package>/settings/publishing/
5. Wait for the user to confirm, then run the release workflow above.
