
The repo contains the source code for the [Making Python 100x faster with less than 100 lines of Rust](https://ohadravid.github.io/posts/2023-03-rusty-python/) blog post (there's a [copy](./rusty_python.md) in this repo as well).

It's a demo library in Python that we can converts parts of to Rust, to improve its performance.

Here's a table summarizing the perf gains:

| Version                                                      | Avg time per iteration (ms)  | Multiplier | 
|--------------------------------------------------------------|------------------------------|------------|
| v1 - Baseline implementation (Python)                        | 293.41                       | 1x         |
| v2 - Naive line-to-line translation of `find_close_polygons` | 23.44                        | 12.50x     |
| v3 - `Polygon` implementation in Rust                        | 6.29                         | 46.53x     |
| v4 - Optimized allocations                                   | 2.90                         | 101.16x    |

There's also a "v1.5" version which is 6x faster, and uses "vectorizing" (doing more of the work directly in numpy).
This version is much harder to optimize further.

## Setup

The code should work on all supported platforms (Python & Rust),
but `--native` profiling is only supported by `py-spy` on x86 Linux and Windows.

(macOS / Arm Linux can still generate profiles viewing just the Python code.)

For example, to setup with conda, run:

```bash
conda install numpy pytest
```

Then install `py-spy`:

```bash
pip install py-spy
```

For the complete Rust setup, you'll need Rust (use [rustup](https://rustup.rs/)) and to also `pip install maturin`.

To build the native extension, run:

```bash
(cd poly_match_rs && maturin develop --release)
```

## Exploring more optimizations

There are a few more optimizations we can try.

For example, we could use `(dx^2 + dy^2) < dist^2` (calculating `dist^2` only once, saving a `sqrt`).
According to the profiler outputs, can we expect this to make a big difference?

We could build a Rust bench using [criterion](https://github.com/bheisler/criterion.rs),
which would require us to "split" the Python & pure-Rust parts (Maybe using `AsRef<Polygon>`).

We could also try to convert the list of Polygons only once to Rust at the start of the run.
According to the profiler outputs, can we expect this to make a big difference?

## License

This repo is licensed under either of
```text
Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
```
at your option.