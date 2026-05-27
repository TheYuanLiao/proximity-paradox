# Container Setup on HPC (Apptainer)

How to go from the shipped Docker container to a working Apptainer environment on an HPC cluster.

## Prerequisites

- HPC account with a storage allocation
- Access to a compute allocation for interactive jobs (e.g., Slurm `salloc`)
- Apptainer (formerly Singularity) available on the cluster — most HPC centres provide it as a module or system package

## Directory Layout

Choose a project directory on the cluster (e.g., your allocation's storage area). Set up this structure:

```
$PROJECT_DIR/
├── containers/
│   └── mycontainer.sif         # Built from the shipped .tar
├── repo/
│   ├── proximity-paradox/      # This git repo which has /infostop
└── data/                       # GPS data and pipeline outputs
```

The container bind-mounts `repo/` as `/workspace` and `data/` as `/data`, so paths inside the container resolve as:

| Host path | Container path |
|-----------|---------------|
| `$PROJECT_DIR/repo/proximity-paradox/` | `/workspace/proximity-paradox/` |
| `$PROJECT_DIR/repo/infostop/` | `/workspace/infostop/` |
| `$PROJECT_DIR/data/` | `/data/` |

## Step 1: Upload and Build the Container

From your local machine, upload the Docker tar archive:

```bash
scp mycontainer.tar <user>@<hpc-login-node>:$PROJECT_DIR/containers/
```

On the HPC cluster, allocate an interactive node and build. Apptainer needs writable temp space — point its temp directories somewhere with enough room (not `/tmp` on shared login nodes):

```bash
# Allocate an interactive node (adjust account, cores, time to your cluster)
salloc -A <your-account> -n 2 -t 04:00:00
srun --pty bash

# Set temp directories (important — default /tmp may be too small or read-only)
export APPTAINER_TMPDIR=$PROJECT_DIR/tmp/apptainer-tmp
export APPTAINER_CACHEDIR=$PROJECT_DIR/tmp/apptainer-cache
export TMPDIR=$PROJECT_DIR/tmp/apptainer-tmp
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

cd $PROJECT_DIR/containers
apptainer build mycontainer.sif docker-archive://mycontainer.tar
```

The build takes ~10–20 minutes. The resulting `.sif` file is read-only and portable.

## Step 2: Clone the Repositories

```bash
cd $PROJECT_DIR/repo
git clone <proximity-paradox-repo-url> proximity-paradox
```

The infostop repo must sit at `$PROJECT_DIR/repo/infostop/` (not inside `proximity-paradox/`), because the container's editable install expects it at `/workspace/infostop/`.

## Step 3: Fix Infostop — Compile `cpputils` Extension

The shipped container has `infostop` installed in **editable mode** (`pip install -e .`), meaning it doesn't bundle the Python source or compiled C++ extension. Instead, it points to `/workspace/infostop/` at runtime via a path finder. Two things must be present on the host for this to work:

1. The `infostop/` Python package source (provided by the git clone above)
2. The compiled `cpputils` C++ extension (`.so` file)

### Why This Is Needed

Infostop uses a C++ module (`cpputils`) built with pybind11 for fast spatial computations. The editable install references it via a path mapping:

```
cpputils → /workspace/infostop/cpputils
```

But the cloned repo only has the C++ source (`cpputils/main.cpp`), not the compiled shared object. You must compile it inside the container.

### Compile `cpputils`

Start a shell inside the container:

```bash
apptainer exec \
  --bind $PROJECT_DIR/repo:/workspace \
  --bind $PROJECT_DIR/data:/data \
  $PROJECT_DIR/containers/mycontainer.sif \
  bash
```

Inside the container:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate proxi

cd /workspace/infostop
pip install -e . --no-deps
```

This compiles `cpputils/main.cpp` into `cpputils.cpython-311-x86_64-linux-gnu.so` in the infostop directory. The `--no-deps` flag avoids reinstalling dependencies that are already in the container.

Verify it worked:

```bash
python -c "import cpputils; print('cpputils OK')"
python -c "from infostop import Infostop; print('infostop OK')"
```

### If `pip install -e .` Fails

The compilation requires `pybind11` and a C++ compiler. Both should be present in the shipped container. If the build fails:

- **Missing pybind11**: `pip install pybind11` inside the container, then retry
- **Missing compiler**: Check `which g++` — if absent, the container image may need rebuilding with `build-essential`

As a fallback, you can compile manually:

```bash
cd /workspace/infostop
python setup.py build_ext --inplace
```

This produces the same `.so` file.

## Step 4: Verify the Environment

```bash
apptainer exec \
  --bind $PROJECT_DIR/repo:/workspace \
  --bind $PROJECT_DIR/data:/data \
  $PROJECT_DIR/containers/mycontainer.sif \
  bash -c '
    eval "$(micromamba shell hook --shell bash)" &&
    micromamba activate proxi &&
    python -c "
import infostop; print(f\"infostop {infostop.__version__}\")
import cpputils; print(\"cpputils OK\")
import pyspark; print(f\"pyspark {pyspark.__version__}\")
import pandas; print(f\"pandas {pandas.__version__}\")
import numpy; print(f\"numpy {numpy.__version__}\")
"'
```

## Running Pipeline Scripts

Allocate resources and launch inside the container:

```bash
# Adjust account, cores, memory, and time to your cluster and dataset size
salloc -A <your-account> -n 20 --mem=64G -t 24:00:00
srun --pty bash

apptainer exec \
  --bind $PROJECT_DIR/repo:/workspace \
  --bind $PROJECT_DIR/data:/data \
  $PROJECT_DIR/containers/mycontainer.sif \
  bash
```

Inside the container:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate proxi

# Example: run stop detection for Sweden
python /workspace/proximity-paradox/src/stop_detection.py \
  --country sweden \
  --input-dir /data/mobile/se_2024/format_parquet \
  --output-dir /data/mobile/se_2024/stops \
  --all --resume --cores 20 --memory 56g
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'infostop'`

The editable install path finder expects infostop source at `/workspace/infostop/`. Make sure:
- The bind mount maps `$PROJECT_DIR/repo` to `/workspace`
- The infostop repo is cloned at `$PROJECT_DIR/repo/infostop/`

### `ModuleNotFoundError: No module named 'cpputils'`

The compiled `.so` file is missing. Follow Step 3 to compile it. Check that `cpputils.cpython-311-x86_64-linux-gnu.so` exists in `$PROJECT_DIR/repo/infostop/`.

### `ModuleNotFoundError: No module named 'pydantic'`

The pipeline scripts (`stop_detection.py`, `device_logging.py`) use an inline YAML config loader instead of pydantic, so pydantic is not required. If other scripts import `config/schema.py` and fail, either install pydantic (`pip install pydantic`) or modify the import.

### Spark driver memory

With a 64 GB memory allocation, set Spark driver memory to **56g** (not 64g). The remaining ~8 GB covers OS, JVM overhead, and Apptainer runtime. Going higher risks OOM kills. Scale proportionally for other allocation sizes.

### Working directory warning

You may see: `WARNING: Error changing the container working directory`. This is harmless — it happens when the container's default working directory doesn't exist inside the container. The scripts use absolute paths.

### Apptainer not found

On some clusters Apptainer is available as a module:
```bash
module spider apptainer    # or: module spider singularity
module load apptainer
```
