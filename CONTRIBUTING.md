# Contributing to LLM Server

Thank you for your interest in contributing to **LLM Server**!  
This project aims to provide a production-style LLM API gateway with observability, authentication, quotas, caching, and multi-model orchestration. Contributions that improve stability, clarity, and maintainability are always welcome.

---

## 1. Code of Conduct

By participating in this project, you agree to uphold the standards in the included `CODE_OF_CONDUCT.md`.  
Respectful communication and collaboration help maintain a productive environment.

---

## 2. How to Contribute

There are many ways to contribute:

- Reporting bugs
- Improving documentation
- Enhancing observability
- Adding new metrics or tests
- Refactoring code for clarity or performance
- Submitting new features (please open an issue first)

If you're unsure whether something fits, **open an issue**—discussion before implementation is encouraged.

---

## 3. Development Environment Setup

### 3.1 Requirements

- Python 3.11+
- `uv` package manager  
- Docker + Docker Compose
- `make`

### 3.2 Clone the Repository

```bash
git clone https://github.com/<your-org>/llm-server.git
cd llm-server
```

### 3.3 Install Dependencies (Local Development Mode)

```bash
uv sync --extra cpu
```

This installs all development extras including test dependencies.

### 3.4 Local LLM Mode (MPS / CPU)

Local mode runs the API **on the host**, while Docker runs Postgres, Redis, Prometheus, Grafana, and Nginx.

```bash
cp .env.example .env.local
make dev-local
```

In another terminal:

```bash
make api-local
```

### 3.5 Full Containerized Mode

```bash
cp .env.example .env
make up
```

Seed an admin API key:

```bash
make seed-key API_KEY=$(openssl rand -hex 24)
```

---

## 4. Running Tests

The test suite uses **pytest** and includes API, rate-limiting, auth, caching, quota, and health tests.

To run all tests:

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/test_generate.py
```

Tests must pass before any PR can be merged.

---

## 5. Style Guidelines

### 5.1 Python Code

- Follow **PEP 8** style.
- Prefer descriptive function names.
- Use `mypy`-friendly type hints.
- Avoid unnecessary abstractions.
- Ensure imports are sorted (`isort`) and formatted (`black`).

### 5.2 API Endpoints

Stable endpoints **must not** introduce breaking changes without discussion.  
Experimental endpoints may evolve quickly but still require documentation.

### 5.3 Documentation

All new features must include:

- Updated README sections
- Updated docs in `docs/`
- Examples if applicable

Documentation is treated as a first-class contribution.

---

## 6. Git Workflow

### 6.1 Branching Model

- `main` — stable releases, production-ready
- Feature branches — `feature/<short-description>`
- Bugfix branches — `fix/<short-description>`

### 6.2 Commit Messages

Use conventional commits when possible:

```
feat: add redis hot-cache layer
fix: correct rpm limit behavior
docs: improve architecture explanation
refactor: simplify multimodel registry
test: extend authentication coverage
```

### 6.3 Pull Requests

Before opening a PR:

- Ensure tests pass
- Ensure lint passes
- Ensure docs are updated
- Describe *what* changed and *why*

PRs should be **small, focused, and easy to review**.

---

## 7. Adding New Features

For significant feature work:

1. **Open an issue** describing the proposed change.
2. Provide:
   - Rationale
   - High-level design
   - API changes (if any)
3. Wait for maintainer approval before implementing.

This ensures alignment with project direction.

---

## 8. Reporting Bugs

When opening a bug report, please include:

- Reproduction steps
- Logs or stack traces
- API responses
- LLM Server version
- Deployment mode (local vs docker)
- Your `.env` (scrub sensitive values)

---

## 9. Security Policy

**Do NOT report vulnerabilities publicly.**

Please email:

`<your-security-email>@protonmail.com`

Include:

- Description of the issue
- Impact assessment
- Steps to reproduce

Responsible disclosure is appreciated.

---

## 10. License

By contributing, you agree that your contributions will be licensed under the **MIT License** used by this project.

---

Thank you for helping make **LLM Server** better and more reliable!