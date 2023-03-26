
The repo contains the source code for the [Making Python 100x faster with less than 100 lines of Rust](https://ohadravid.github.io/posts/2023-03-rusty-python/) blog post (there's a [copy](./rusty_python.md) in this repo as well).

It's a demo library in Python that we can converts parts of to Rust, to improve its performance.

Here's a table summarizing the perf gains:


| Version                                                      | Avg time per iteration (ms)  | Multiplier | 
|--------------------------------------------------------------|------------------------------|------------|
| v1 - Baseline implementation (Python)                        | 293.41                       | 1x         |
| v2 - Naive line-to-line translation of `find_close_polygons` | 23.44                        | 12.50x     |
| v3 - `Polygon` implementation in Rust                        | 6.29                         | 46.53x     |
| v4 - Optimized allocations                                   | 2.90                         | 101.16x    |
| v1.5 - Example of "vectorizing" using Numpy                  | 48.82                        | 6x         |

The code should work on all supported platforms (Python & Rust),
but `--native` profiling is only supported by `py-spy` in x86 Linux or Windows.

(macOS / Arm Linux can still generate profiles viewing just the Python code.)

For example, to setup with conda, run:

```bash
conda install numpy pytest
```

Then install `py-spy`:

```bash
pip install py-spy
```

This repo is licensed under either of
```text
Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
```
at your option.