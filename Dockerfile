#
# PyBNF container image -- legacy BNG2.pl backend.
#
# This image bundles BioNetGen's BNG2.pl backend (the open-source rule-based
# modeling engine) together with PyBNF and its public runtime dependencies.
#
# It deliberately does NOT install bngsim, PyBNF's modern simulation backend:
# bngsim is a private, macOS-only wheel with no public Linux build, so it cannot
# be installed in a Linux container today (issue #414). PyBNF imports cleanly
# without bngsim and falls back to the legacy BNG2.pl path. When bngsim ships
# public Linux wheels, a bngsim-backed image can replace this one.
#
# Build (run inside the PyBNF directory):
#     docker build -t pybnf .
#
# Run with your examples directory mounted:
#     docker run -it --rm -v $(pwd)/examples:/project/examples pybnf
#
# And then inside the image:
#     cd examples/demo
#     pybnf -c demo_bng.conf
#

ARG PYTHON_VERSION=3.13
ARG BIONETGEN_VERSION=BioNetGen-2.9.3


### BioNetGen: fetch the prebuilt Linux release (no from-source compile)
FROM python:${PYTHON_VERSION}-slim AS bionetgen
ARG BIONETGEN_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src
# Use the official prebuilt Linux tarball (BNG2.pl plus its bundled
# run_network/NFsim binaries) instead of compiling from source. This mirrors the
# CI provisioning in .github/actions/setup-pybnf/action.yml and keeps the build
# fast and robust (no autoconf/cmake/ninja toolchain needed).
RUN curl -fsSL \
      "https://github.com/RuleWorld/bionetgen/releases/download/${BIONETGEN_VERSION}/${BIONETGEN_VERSION}-linux.tar.gz" \
      -o bionetgen.tar.gz && \
    tar -xzf bionetgen.tar.gz && \
    rm bionetgen.tar.gz


### PyBNF wheel build container
FROM python:${PYTHON_VERSION}-slim AS pybnf-build

WORKDIR /usr/src/PyBNF
COPY pyproject.toml README.md LICENSE ./
COPY pybnf ./pybnf

RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .


### Runtime container
FROM python:${PYTHON_VERSION}-slim

ARG BIONETGEN_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    perl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=bionetgen /usr/src/${BIONETGEN_VERSION} /usr/BioNetGen
COPY --from=pybnf-build /wheels /tmp/wheels

# Install PyBNF's PUBLIC runtime dependencies explicitly, then install the wheel
# with --no-deps. We must not let pip re-resolve PyBNF's dependency set here: it
# pins bngsim>=0.5.0,<1, a private macOS-only wheel with no public Linux build,
# which would make the install fail (issue #414). This list mirrors
# pyproject.toml's [project.dependencies] minus bngsim; keep it in sync.
RUN python -m pip install --no-cache-dir \
      'dask>=2024.1.0' \
      'distributed>=2024.1.0' \
      'libroadrunner>=1.6.0' \
      'msgpack>=0.6.2' \
      'numpy>=1.24' \
      'paramiko>=2.7' \
      'pydantic>=2,<3' \
      'pyparsing>=2.4,<4' \
      'scipy>=1.10' \
      'tornado>=6.1' && \
    python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

ENV BNGPATH=/usr/BioNetGen
ENV PATH="${BNGPATH}:${PATH}"

WORKDIR /project

CMD ["bash"]
