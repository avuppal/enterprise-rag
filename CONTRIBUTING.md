# Contributing to enterprise-rag

Thank you for your interest in contributing!  
This project follows a small set of conventions to keep the codebase consistent.

---

## Getting Started

```bash
# Fork & clone
git clone https://github.com/<your-fork>/enterprise-rag.git
cd enterprise-rag

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies + dev extras
pip install -r requirements.txt
pip install pytest pytest-cov ruff black
```

---

## Development Workflow

1. **Branch** off `main` with a descriptive name:
   ```bash
   git checkout -b feat/mmr-improvements
   ```

2. **Write code** — keep functions small and focused.

3. **Write tests** — every new function must have at least one test.
   All tests must pass without any external services:
   ```bash
   pytest -m "not integration"
   ```

4. **Format & lint:**
   ```bash
   ruff check src/ tests/
   black src/ tests/
   ```

5. **Open a PR** against `main` with a clear description.

---

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings for every public class and function (NumPy style)
- No hardcoded API keys or model names in library code — always injectable
- Prefer `List`, `Dict`, `Optional` from `__future__ annotations` for 3.10 compat

---

## Testing Standards

- **CPU-only**: all tests must pass without GPU, internet, or external services
- **Mock external deps** (ChromaDB, sentence-transformers) via `sys.modules` injection
- Tag integration tests with `@pytest.mark.integration` so they can be excluded
- Aim for ≥ 90 % line coverage on new modules

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add MMR diversity selector
fix: BM25 IDF calculation for empty corpus
docs: add Mermaid diagram for eval framework
test: add NDCG edge-case for k > len(retrieved)
refactor: extract _tokenize helper into shared util
```

---

## Reporting Issues

Open a GitHub issue with:
- Python version
- Reproduction steps (minimal code snippet)
- Expected vs. actual behaviour
- Stack trace if applicable

---

## Questions?

Open a Discussion on GitHub — happy to help!
