ARCH_STRING=$(rocminfo | grep 'Name:' | grep -o 'gfx9[0-9a-z]*' | head -n 1)
docker build --build-arg GPU_ARCH=$ARCH_STRING -f amd_sky.ubuntu.amd.Dockerfile -t motif_image .
