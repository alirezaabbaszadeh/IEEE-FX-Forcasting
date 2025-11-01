FROM continuumio/miniconda3:23.5.2-0

WORKDIR /workspace

COPY environment.yml ./
RUN conda env create -f environment.yml \
    && conda clean -afy

SHELL ["conda", "run", "-n", "ieee-fx", "/bin/bash", "-c"]

COPY . ./

ENV PYTHONPATH=/workspace

CMD ["conda", "run", "-n", "ieee-fx", "python", "-m", "src.cli"]
