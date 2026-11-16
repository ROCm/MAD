# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch-training:v25.5
FROM $BASE_DOCKER

# Additional step for base images that ship setup_pytorch_env.
# Guard on the setup command itself rather than on the workspace directory, so this
# same Dockerfile still builds against a plain upstream base, which does not ship it.
# The --mad flag installs into the image's existing environment instead of creating a
# venv. There is no /opt/venv-py<X.Y> to link, so this block has no symlink step and --
# importantly -- no "ENV PATH=/opt/venv/bin:${PATH}" line after it. Adding one would not
# fail the build (ENV never touches the filesystem, and the shell silently skips PATH
# entries that do not exist), it would just be dead PATH that hides which interpreter is
# actually in use. Leave it out.
RUN if command -v setup_pytorch_env >/dev/null 2>&1; then \
        echo "setup_pytorch_env found. Running env setup..."; \
        setup_pytorch_env --mad || { \
            echo "Setup failed."; \
            exit 1; \
        }; \
        echo "Setup successful."; \
        for pair in python:python3 pip:pip3; do \
            name="${pair%%:*}"; target="${pair#*:}"; \
            if ! command -v "$name" >/dev/null 2>&1; then \
                ln -s "$(command -v "$target")" "/usr/local/bin/$name" || { \
                    echo "Failed to link $name -> $target."; \
                    exit 1; \
                }; \
                echo "Linked $name -> $(command -v "$target")"; \
            fi; \
        done; \
    else \
        echo "setup_pytorch_env is not on PATH."; \
        echo "Nothing to set up."; \
    fi

# Pin numpy to < 2.0.0 and pandas<3.0
RUN pip install --upgrade numpy==1.26.4 scipy numba "pandas<3.0"
RUN apt-get update && apt-get install -y zip unzip