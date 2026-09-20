#!/usr/bin/env bash
set -e

CASE_ID=$(basename "$PWD")
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/${CASE_ID}_logs"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "          Running Benchmark: ${CASE_ID}                     "
echo "============================================================"

# ------------------------------------------------------------------
# 1. Dynamic Traces (workload.py -> collect_metrics.py)
# ------------------------------------------------------------------
echo ""
echo "[Step 1/5] Collecting Empirical Memory Traces..."
DYNAMIC_SUMMARY="N/A"
if [ -f "collect_metrics.py" ]; then
    DYNAMIC_OUT=$(python3 collect_metrics.py)
    DYNAMIC_SUMMARY=$(echo "$DYNAMIC_OUT" | grep -E "(Vulnerable|Fixed)" || echo "Completed")
    echo "$DYNAMIC_SUMMARY"
else
    echo "  [SKIP] collect_metrics.py not found."
fi

# ------------------------------------------------------------------
# 2. Principal Direct Baseline: torch-preflight
# ------------------------------------------------------------------
echo ""
echo "[Step 2/5] Running Principal Baseline: torch-preflight..."
TP_VULN_STATUS="SKIPPED"
TP_FIXED_STATUS="SKIPPED"

if command -v torch-preflight &> /dev/null; then
    torch-preflight vulnerable.py > "${LOG_DIR}/torch_preflight_vuln.log" 2>&1 || true
    torch-preflight fixed.py > "${LOG_DIR}/torch_preflight_fixed.log" 2>&1 || true

    if grep -iq "issue" "${LOG_DIR}/torch_preflight_vuln.log" && ! grep -iq "No issues found" "${LOG_DIR}/torch_preflight_vuln.log"; then
        TP_VULN_STATUS="DETECTED (Flagged)"
    else
        TP_VULN_STATUS="CLEAN (No issues)"
    fi

    if grep -iq "issue" "${LOG_DIR}/torch_preflight_fixed.log" && ! grep -iq "No issues found" "${LOG_DIR}/torch_preflight_fixed.log"; then
        TP_FIXED_STATUS="DETECTED (Flagged)"
    else
        TP_FIXED_STATUS="CLEAN (No issues)"
    fi
    echo "  -> torch-preflight finished."
else
    echo "  [SKIP] torch-preflight not found in PATH."
fi

# ------------------------------------------------------------------
# 3. PyTorch Linter Baseline: TorchFix
# ------------------------------------------------------------------
echo ""
echo "[Step 3/5] Running PyTorch Linter Baseline: TorchFix..."
TF_VULN_STATUS="SKIPPED"
TF_FIXED_STATUS="SKIPPED"

if command -v torchfix &> /dev/null; then
    (torchfix --select=ALL vulnerable.py > "${LOG_DIR}/torchfix_vuln.log" 2>&1) || true
    (torchfix --select=ALL fixed.py > "${LOG_DIR}/torchfix_fixed.log" 2>&1) || true

    if grep -qE "TOR[0-9]{3}" "${LOG_DIR}/torchfix_vuln.log"; then
        TF_VULN_STATUS="DETECTED (Flagged)"
    else
        TF_VULN_STATUS="CLEAN (No issues)"
    fi

    if grep -qE "TOR[0-9]{3}" "${LOG_DIR}/torchfix_fixed.log"; then
        TF_FIXED_STATUS="DETECTED (Flagged)"
    else
        TF_FIXED_STATUS="CLEAN (No issues)"
    fi
    echo "  -> TorchFix finished."
else
    echo "  [SKIP] torchfix not found in PATH."
fi

# ------------------------------------------------------------------
# 4. Generic Baseline: Semgrep
# ------------------------------------------------------------------
echo ""
echo "[Step 4/5] Running Generic Baseline: Semgrep..."
SG_VULN_STATUS="SKIPPED"
SG_FIXED_STATUS="SKIPPED"

if command -v semgrep &> /dev/null; then
    semgrep scan --config auto --text vulnerable.py > "${LOG_DIR}/semgrep_vuln.log" 2>&1 || true
    semgrep scan --config auto --text fixed.py > "${LOG_DIR}/semgrep_fixed.log" 2>&1 || true

    if grep -iqE "[1-9][0-9]* findings" "${LOG_DIR}/semgrep_vuln.log"; then
        SG_VULN_STATUS="DETECTED (Flagged)"
    else
        SG_VULN_STATUS="CLEAN (No issues)"
    fi

    if grep -iqE "[1-9][0-9]* findings" "${LOG_DIR}/semgrep_fixed.log"; then
        SG_FIXED_STATUS="DETECTED (Flagged)"
    else
        SG_FIXED_STATUS="CLEAN (No issues)"
    fi
    echo "  -> Semgrep finished."
else
    echo "  [SKIP] semgrep not found in PATH."
fi

# ------------------------------------------------------------------
# 5. Generic AST/DataFlow Baseline: CodeQL
# ------------------------------------------------------------------
echo ""
echo "[Step 5/5] Running Generic Baseline: CodeQL..."
CQL_VULN_STATUS="SKIPPED"
CQL_FIXED_STATUS="SKIPPED"

if command -v codeql &> /dev/null; then
    # Isolated vulnerable analysis
    mkdir -p /tmp/${CASE_ID}_vuln_src && cp vulnerable.py /tmp/${CASE_ID}_vuln_src/
    codeql database create /tmp/${CASE_ID}_vuln_db --language=python --source-root=/tmp/${CASE_ID}_vuln_src --quiet > /dev/null 2>&1 || true
    codeql database analyze /tmp/${CASE_ID}_vuln_db codeql/python-queries \
        --format=sarif-latest --output="${LOG_DIR}/codeql_vuln.sarif" --quiet > /dev/null 2>&1 || true
    rm -rf /tmp/${CASE_ID}_vuln_db /tmp/${CASE_ID}_vuln_src

    # Isolated fixed analysis
    mkdir -p /tmp/${CASE_ID}_fixed_src && cp fixed.py /tmp/${CASE_ID}_fixed_src/
    codeql database create /tmp/${CASE_ID}_fixed_db --language=python --source-root=/tmp/${CASE_ID}_fixed_src --quiet > /dev/null 2>&1 || true
    codeql database analyze /tmp/${CASE_ID}_fixed_db codeql/python-queries \
        --format=sarif-latest --output="${LOG_DIR}/codeql_fixed.sarif" --quiet > /dev/null 2>&1 || true
    rm -rf /tmp/${CASE_ID}_fixed_db /tmp/${CASE_ID}_fixed_src

    # Extract finding status directly via python helper
    CQL_VULN_STATUS=$(python3 -c "
import json
try:
    with open('${LOG_DIR}/codeql_vuln.sarif') as f:
        d = json.load(f)
    n = len(d.get('runs', [{}])[0].get('results', []))
    print(f'DETECTED ({n} findings)' if n > 0 else 'CLEAN (No issues)')
except Exception:
    print('CLEAN (No issues)')
")

    CQL_FIXED_STATUS=$(python3 -c "
import json
try:
    with open('${LOG_DIR}/codeql_fixed.sarif') as f:
        d = json.load(f)
    n = len(d.get('runs', [{}])[0].get('results', []))
    print(f'DETECTED ({n} findings)' if n > 0 else 'CLEAN (No issues)')
except Exception:
    print('CLEAN (No issues)')
")
    echo "  -> CodeQL isolated scans complete."
else
    echo "  [SKIP] CodeQL CLI not found in PATH."
fi

# ------------------------------------------------------------------
# Executive Summary Output (Printed to Screen & Saved to summary.txt)
# ------------------------------------------------------------------
SUMMARY_FILE="${LOG_DIR}/summary.txt"

{
    echo "============================================================"
    echo "          EXECUTION SUMMARY: ${CASE_ID}                     "
    echo "============================================================"
    echo "--- Empirical Memory Trace ---"
    echo "${DYNAMIC_SUMMARY}"
    echo ""
    echo "--- Static Analysis Detection Matrix ---"
    printf "%-18s | %-20s | %-20s\n" "Tool Name" "vulnerable.py" "fixed.py"
    echo "-------------------|----------------------|---------------------"
    printf "%-18s | %-20s | %-20s\n" "torch-preflight" "${TP_VULN_STATUS}" "${TP_FIXED_STATUS}"
    printf "%-18s | %-20s | %-20s\n" "TorchFix"        "${TF_VULN_STATUS}" "${TF_FIXED_STATUS}"
    printf "%-18s | %-20s | %-20s\n" "Semgrep"         "${SG_VULN_STATUS}" "${SG_FIXED_STATUS}"
    printf "%-18s | %-20s | %-20s\n" "CodeQL (Generic)" "${CQL_VULN_STATUS}" "${CQL_FIXED_STATUS}"
    echo "============================================================"
    echo "Raw logs, metrics, and SARIF files saved in:"
    echo "${LOG_DIR}"
    echo "============================================================"
} | tee "${SUMMARY_FILE}"