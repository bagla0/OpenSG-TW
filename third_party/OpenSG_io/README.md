# `third_party/OpenSG_io` — vendored converter

The Python package in `opensg_io/` and the helpers in `scripts/` are vendored verbatim from
**[OpenSG_io](https://github.com/bagla0/OpenSG_io)** (commit `3ec56a9`) — the companion windIO / PreVABS /
OpenFAST → OpenSG cross-section converter. OpenSG-TW uses it to build the IEA-22 blade cross-section YAMLs
straight from the source **windIO** file (see {doc}`../../docs/tutorials/iea22_windio_to_timo` and
{doc}`../../docs/tutorials/iea22_full_blade`).

| Path | What |
|---|---|
| `opensg_io/converter.py` | `load_blade`, `build_cross_section`, `emit_opensg_yaml`, `emit_prevabs` |
| `opensg_io/prevabs_xml.py` | PreVABS XML emit + `to_solid` / `to_shell` wrappers |
| `scripts/convert_blade.py` | windIO → per-station 1-D shell + (with `--solid`) 2-D solid YAML |
| `scripts/convert_sg_to_yaml.py` | PreVABS `.sg` → OpenSG 2-D-solid YAML |

**PreVABS is *not* vendored** — it is a separate ~300 MB binary (itself a submodule of OpenSG_io pointing at
[wenbinyugroup/prevabs](https://github.com/wenbinyugroup/prevabs)). The **windIO → 1-D-shell-YAML** step runs
with `opensg_io` alone (no binary); generating the **2-D-solid** YAML additionally requires a PreVABS install.

Upstream: <https://github.com/bagla0/OpenSG_io> · License: see `LICENSE`.
