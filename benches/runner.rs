// Benchmarks for the agent runner.
//
// Run with: `cargo bench`

use criterion::{criterion_group, criterion_main, Criterion};

fn runner_benchmark(c: &mut Criterion) {
    // TODO: Add benchmarks once Runner is implemented.
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // Placeholder benchmark.
            std::hint::black_box(42)
        });
    });
}

criterion_group!(benches, runner_benchmark);
criterion_main!(benches);
