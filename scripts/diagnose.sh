#!/usr/bin/env bash
set -e

# 1. Create a clean output directory
mkdir -p diagnostics
echo "Starting framework diagnostics..."

# 2. Map Architecture & Dependencies (JSON)
echo "-> Running pydeps (Architecture)..."
uv tool run pydeps hoid --show-deps --noshow --no-output > diagnostics/01_architecture.json 2>&1

# 3. Deep Logic & Anti-Pattern Scan (ALL RULES)
echo "-> Running ruff (Logic & Anti-Patterns)..."
uv tool run ruff check hoid/ --select ALL --statistics > diagnostics/02_logic_errors.txt 2>&1 || true

# 4. Security Vulnerabilities
echo "-> Running bandit (Security)..."
uv tool run bandit -r hoid/ -ll > diagnostics/03_security_risks.txt 2>&1 || true

# 5. Code Complexity (Cyclomatic Scores)
echo "-> Running radon (Complexity)..."
uv tool run radon cc hoid/ -s -a > diagnostics/04_complexity.txt 2>&1

# 6. Type verification & Circular Imports
echo "-> Running mypy (Types)..."
uv tool run mypy hoid/ --ignore-missing-imports > diagnostics/05_type_errors.txt 2>&1 || true

# 7. Dependency Hygiene (Unused/Missing packages)
echo "-> Running deptry (Dependencies)..."
# 2>&1 is absolutely critical here to capture the deptry findings
uv tool run deptry . --no-ansi > diagnostics/06_dependencies.txt 2>&1 || true

# 8. Runtime Dead Code Map
echo "-> Running pytest coverage (Dead Code)..."
uv run --with pytest-cov pytest --cov=hoid --cov-report=term-missing tests/unit > diagnostics/07_coverage.txt 2>&1 || true

echo "========================================"
echo "Done! All diagnostic data saved to './diagnostics'."
echo "Upload this folder to your LLM for a complete architectural teardown."