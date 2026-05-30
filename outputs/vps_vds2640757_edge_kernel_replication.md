# HYP-002 VPS Replication Result

Status: second-machine pass

Host:

```text
vds2640757
```

Environment:

```text
platform: Linux-6.8.0-110-generic-x86_64-with-glibc2.39
machine: x86_64
python: 3.12.3
numpy: 1.26.4
compiler: g++ 13.3.0
```

Environment caveat:

```text
NumPy 2.4.6 failed because the VPS CPU does not support the X86_V2 baseline
required by that wheel. The replication used numpy==1.26.4.
```

## Results

| Backend | Memory bytes | 1000 passes ms | Mismatches |
|---|---:|---:|---:|
| NumPy float32 modular | 8452 | 323.063606 | 0 |
| NumPy uint32 GF(137) reference | 2113 | 767.871869 | 0 |
| C++ portable O3 GF(137) | 2113 | 241.660830 | 0 |
| C++ native O3 GF(137) | 2113 | 279.983902 | 0 |

Summary:

```text
Storage ratio float32/native GF(137): 4.000x
Runtime speedup float32/native GF(137): 1.154x
Mismatch count: 0
pass_storage = true
pass_agreement = true
pass_speed = true
```

Interpretation:

This is a second-machine replication pass for the engineering claim.  The
runtime effect is much smaller than on the local Apple ARM machine, but it
remains positive under the frozen pass condition.  The result does not establish
a fundamental-physics claim.
