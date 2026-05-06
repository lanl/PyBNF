#
# To build Docker image, run inside PyBNF directory:
#     docker build -t pybnf .
#
# If BNGsim is hosted on a private package index, pass the index URL:
#     docker build --build-arg PIP_EXTRA_INDEX_URL=https://... -t pybnf .
#
# Run inside PyBNF to mount examples directory inside the image:
#     docker run -it --rm -v $(pwd)/examples:/project/examples pybnf
#
# And then inside the image:
#     cd examples/demo
#     pybnf -c demo_bng.conf
#

ARG PYTHON_VERSION=3.12
ARG BIONETGEN_VERSION=BioNetGen-2.9.3


### BioNetGen build container
FROM python:${PYTHON_VERSION}-slim AS bionetgen-build

ARG BIONETGEN_VERSION

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libtool \
    ninja-build \
    perl && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recurse-submodules --branch "${BIONETGEN_VERSION}" https://github.com/RuleWorld/bionetgen.git

WORKDIR /usr/src/bionetgen/bng2
RUN ./make_dist.pl --build


### PyBNF wheel build container
FROM python:${PYTHON_VERSION}-slim AS pybnf-build

WORKDIR /usr/src/PyBNF
COPY pyproject.toml README.md LICENSE ./
COPY pybnf ./pybnf

RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .


### Runtime container
FROM python:${PYTHON_VERSION}-slim

ARG BIONETGEN_VERSION
ARG PIP_EXTRA_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    perl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=bionetgen-build /usr/src/bionetgen/bng2/${BIONETGEN_VERSION} /usr/BioNetGen
COPY --from=pybnf-build /wheels /tmp/wheels

RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

ENV BNGPATH=/usr/BioNetGen
ENV PATH="${BNGPATH}:${PATH}"

WORKDIR /project

CMD ["bash"]
