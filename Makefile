.PHONY: build test lint fmt fmt-check clippy doc bench clean check coverage

# Build
build:
	cargo build

build-release:
	cargo build --release

# Format
fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

# Lint
clippy:
	cargo clippy -- -W clippy::all

lint: clippy

# Test
test:
	cargo test

test-verbose:
	cargo test -- --nocapture

# Benchmarks
bench:
	cargo bench

# Documentation
doc:
	cargo doc --no-deps --open

build-docs:
	mdbook build docs/

serve-docs:
	mdbook serve docs/

# Coverage (requires cargo-tarpaulin)
coverage:
	cargo tarpaulin --out html --output-dir coverage/ --fail-under 85

# Snapshot tests (insta)
snapshots-review:
	cargo insta review

snapshots-accept:
	cargo insta accept

# Clean
clean:
	cargo clean

# Full check (run before committing)
check: fmt-check clippy test
	@echo "All checks passed!"
