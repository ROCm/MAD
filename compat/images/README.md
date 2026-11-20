# Named image specs

Each `<name>.yaml` here is an image spec (the format of `scripts/<dir>/image_spec.yaml`)
that several workloads share. A card selects one through madengine's OmegaConf
interpolation and may override any part of it:

```json
"image_spec": "${images.<name>}"
"image_spec": {"extends": "${images.<name>}", "gpu_arch": "gfx950"}
```

A spec may give its base per GPU architecture (`base: {gfx942: ..., gfx950: ...}`), put
files from `docker/` into the image (`copy`), and run checks after its components
(`finish`). Its components are builds authorized in `compat/rocm.yaml`.
