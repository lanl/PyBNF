#
# To build Docker image, run inside PyBNF directory:
#     docker build -t pybnf .
#
# Run inside PyBNF to mount examples directory inside the image:
#     $ docker run -it --rm -v $(pwd)/examples:/project/examples pybnf
#
# And then inside the image:
#     cd examples/demo
#     pybnf -c demo_bng.conf
#
#

#### Interim build container
FROM continuumio/anaconda3

# Build BioNetGen package from source
WORKDIR /usr/src
RUN apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    cmake \
    libtool \
    ninja-build && \
    git clone https://github.com/RuleWorld/bionetgen.git && \
    cd bionetgen && \
    git submodule init && \
    git submodule update && \
    git checkout BioNetGen-2.3.1 && \
    cd bng2 && \
    perl -i -ne'print unless /isnan/' Network3/src/network.h && \
    ./make_dist.pl --build

# Copy PyBNF source into container
WORKDIR /usr/PyBNF
COPY . .

# Build PyBNF binary wheel
RUN python3 setup.py bdist_wheel

# Build psutil binary wheel
WORKDIR /usr/PyBNF/dist
RUN pip wheel psutil


### Minimal PyBNF Docker container
FROM continuumio/miniconda3

# Copy compiled packages from build container
COPY --from=0 /usr/src/bionetgen/bng2/BioNetGen-2.3.1 /usr/BioNetGen-2.3.1
COPY --from=0 /usr/PyBNF/dist/*.whl /tmp/

# Install Python packages
WORKDIR /tmp
RUN pip install --no-cache-dir *.whl && \
    rm *.whl

# Setup environment
ENV BNGPATH /usr/BioNetGen-2.3.1
ENV PATH $BNGPATH:$PATH

WORKDIR /project

CMD [ "bash" ]

