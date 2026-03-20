# Conda packaging for gcrack

This directory contains a `conda-build` recipe and scripts to:

1. Build locally from this repository.
2. Upload the built package to your Anaconda channel.

## 1) Create a build environment

```bash
conda create -n gcrack-conda-build -c conda-forge python=3.14 conda-build anaconda-client -y
conda activate gcrack-conda-build
```

The recipe builds variants for Python `3.10`, `3.11`, `3.12`, `3.13`, and `3.14`.

## 2) Build locally

From the repository root:

```bash
bash packaging/conda/build_local.sh
```

Artifacts are written under:

```bash
packaging/conda/build-artifacts/<platform>/gcrack-*.conda
```

Optional local installation test:

```bash
conda create -n gcrack-local-test -c file:///Users/administrator/Desktop/repos/gcrack/packaging/conda/build-artifacts -c conda-forge python=3.12 gcrack -y
conda activate gcrack-local-test
python -c "import gcrack; print(gcrack.__name__)"
```

Note: installing a `.conda` file path directly (for example `.../gcrack-2026.01.26-py314_0.conda`) is an explicit spec and can bypass normal dependency solving.
Always install by package name from the local channel directory (`-c file:///.../build-artifacts`) so `run` dependencies are resolved.

You can also use the helper script from repository root:

```bash
bash packaging/conda/test_local_install.sh gcrack-local-test-312 3.12
```

Run all supported Python local-install checks:

```bash
bash packaging/conda/test_local_install_matrix.sh
```

## 3) Upload to your Anaconda channel

Two authentication options are supported by `upload_channel.sh`:

- `ANACONDA_API_TOKEN` environment variable (recommended for automation)
- interactive `anaconda login`

Upload command:

```bash
bash packaging/conda/upload_channel.sh <your-channel>
```

Install from your channel:

```bash
conda install -c <your-channel> -c conda-forge gcrack
```

## 4) Release checklist

For each new release:

1. Update version in `pyproject.toml`.
2. Update `package.version` in `packaging/conda/meta.yaml`.
3. Rebuild with `bash packaging/conda/build_local.sh`.
4. Upload with `bash packaging/conda/upload_channel.sh <your-channel>`.

## 5) GitHub CI/CD for Conda

This repository includes:

- `.github/workflows/conda_package.yml`

What it does:

1. Builds conda packages on `ubuntu-latest`, `macos-15`, and `windows-latest`.
2. Uploads build artifacts to the workflow run.
3. Optionally publishes artifacts to your Anaconda channel.

Note: Windows is configured as non-blocking in CI because some scientific dependencies may be unavailable there depending on channel state.

Trigger modes:

- Manual: `workflow_dispatch` (set input `publish=true` to upload).
- Automatic publish: push a tag like `v2026.01.26`.

Required repository secrets:

- `ANACONDA_API_TOKEN`: token created on anaconda.org.
- `ANACONDA_CHANNEL`: your anaconda username/channel.

After pushing a tag, install from your channel with:

```bash
conda install -c <your-channel> -c conda-forge gcrack
```

## 6) Troubleshooting warnings

If you see this during `conda build`:

- `No numpy version specified in conda_build_config.yaml`:
	- Fixed by `packaging/conda/conda_build_config.yaml` in this repository.
- `Number of parsed outputs does not match detected raw metadata blocks`:
	- Usually a parser warning from templated metadata. The recipe here avoids Jinja output blocks to reduce this.
- `RequestsDependencyWarning ... urllib3/chardet/charset_normalizer`:
	- This comes from your build environment, not `gcrack` itself.
	- Refresh the build env packages, for example:

```bash
conda activate gcrack-conda-build
conda install -c conda-forge "requests>=2.32" "urllib3>=2" "charset-normalizer>=3" -y
```

If dependency solving fails with missing packages, verify your channels expose these recipe names:

- `fenics-dolfinx`
- `fenics-ufl`
- `python-gmsh`
