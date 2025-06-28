# copilot-instructions.md

This file provides guidance to LLMs when working with code in this repository.

## Overview

This is the lapper repository. It will contain code for benchmarking binary
and kary search on CPU and GPU. It uses the Mojo language.

Mojo is a programming language that bridges the gap between research and production
by combining Python syntax and ecosystem with systems programming and metaprogramming
features.

## Essential Commands

### Building the Standard Library

```bash
./stdlib/scripts/build-stdlib.sh
```

This creates a `build/stdlib.mojopkg` file in the repo root.

### Running Tests

```bash
# Run all tests
pixi run t

# View all tests
pixi run t --co

# Run specific tests that match a pattern
pixi run t --filter test_binary_search

# Run test via its main function for print output when the test is passing
pixi run r tests/lapper/cpu/test_naive_binary.mojo
```

Tests are run with `-D ASSERT=all` by default to enable bounds checking in stdlib.

### Running Benchmarks

```bash
# Build CPU benchmarks
pixi run bc

# Run CPU benchmarks
./cpu_bench

# Build CPU benchmarks
pixi run bg

# Run GPU benchmarks
./gpu_bench
```

### Code Formatting

```bash
# Format all Mojo files
pixi run mojo format ./

# Format is automatically applied via pre-commit hooks
```

### Documentation Validation

```bash
pixi run mojo doc --diagnose-missing-doc-strings --validate-doc-strings \
  -o /dev/null stdlib/stdlib/
```

## High-Level Architecture

### Repository Structure

- `lapper/`: Library for k-ary search
    - `cpu`: cpu related code
    - `gpu`: gpu related code
- `benchmarks`: Benchmarking tooling
- `tests`: All tests for the library code, should mirror folder structure.

### Key Development Patterns

#### Test Structure

- Tests should use `def` methods to allow for raising.
- Tests should be named `test_*`, and should be added under the `def main` in each file.

#### Memory Management

- Follow value semantics and ownership conventions
- Use `Reference` types and lifetimes in APIs
- Prefer `AnyType` over `AnyTrivialRegType` (except for MLIR interactions)

## Development Workflow

1. **Branch from `main`**: Always work off the main branch (for nightly builds)
2. **Install Mojo**: Use the latest stable build for development
3. **Use VS Code extension**: Install the Mojo extension
4. **Small PRs**: Keep pull requests under 100 lines when possible
5. **Test your changes**: Run relevant tests before submitting
6. **Format code**: Ensure code passes `mojo format`
7. **Document APIs**: Add docstrings following the style guide

## Critical Notes

- **Do NOT** commit secrets or API keys
- **Do NOT** break existing APIs without discussion
- **Do NOT** add dependencies without discussion
- Prefer using Batch tool for multiple file operations to reduce context usage
- When making multiple bash calls, use Batch to run them in parallel

## Performance Considerations

- Performance improvements must include benchmarks
- Don't sacrifice readability for minor performance gains
- Use the benchmarking infrastructure to track regressions

## Platform Support

- Linux x86_64 and aarch64
- macOS ARM64
- Windows is not currently supported

## Internal APIs

The following are private/internal APIs without backward compatibility
guarantees:

- MLIR dialects (`pop`, `kgen`, `lit`)
- Compiler runtime features (prefixed with `KGEN_CompilerRT_`)

## Contribution Guidelines

- Bug fixes should include reproducing tests
- New features should align with the roadmap
- All code must have corresponding tests
- Follow the coding style guide strictly
- Use pre-commit hooks for automatic formatting

## LLM-friendly Documentation

- Docs index: <https://docs.modular.com/llms.txt>
- Mojo API docs: <https://docs.modular.com/llms-mojo.txt>
- Python API docs: <https://docs.modular.com/llms-python.txt>
- Comprehensive docs: <https://docs.modular.com/llms-full.txt>

## Mojo Language Memories

- it's `mut` not `inout`