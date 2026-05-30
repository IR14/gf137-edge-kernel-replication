# HYP-002 GitHub Actions Replication Result

Status: CI pass

Artifact source:

```text
Edge Kernel Replication.zip
```

Environment:

```text
platform: Linux-6.17.0-1015-azure-x86_64-with-glibc2.39
machine: x86_64
python: 3.12.13
numpy: 1.26.4
compiler: Ubuntu clang 18.1.3
```

Results:

| Backend | Memory bytes | 1000 passes ms | Mismatches |
|---|---:|---:|---:|
| NumPy float32 modular | 8452 | 159.561371 | 0 |
| NumPy uint32 GF(137) reference | 2113 | 453.837748 | 0 |
| C++ portable O3 GF(137) | 2113 | 64.945522 | 0 |
| C++ native O3 GF(137) | 2113 | 62.582277 | 0 |

Summary:

```text
Storage ratio float32/native GF(137): 4.000x
Runtime speedup float32/native GF(137): 2.550x
Mismatch count: 0
pass_storage = true
pass_agreement = true
pass_speed = true
```

Interpretation:

This is a CI replication pass on a hosted Ubuntu x86_64 runner.  Together with
the local Apple ARM run and the manual VPS Linux/x86_64 run, it supports the
engineering replication claim under the frozen HYP-002 gate.
