#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv_baselines"

mkdir -p "${ENV_DIR}"

create_env() {
  local env_name="$1"
  local py_bin="$2"
  local req_file="$3"
  local env_path="${ENV_DIR}/${env_name}"

  if [[ ! -d "${env_path}" ]]; then
    "${py_bin}" -m venv "${env_path}"
  fi

  "${env_path}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${env_path}/bin/pip" install -r "${req_file}"
}

REQ_FOUNDATION="${ENV_DIR}/requirements.foundation.txt"
REQ_UNI2TS="${ENV_DIR}/requirements.uni2ts.txt"
REQ_MOMENT="${ENV_DIR}/requirements.moment.txt"
REQ_LEGACY="${ENV_DIR}/requirements.legacy.txt"

cat > "${REQ_FOUNDATION}" <<'EOF'
numpy>=1.26,<1.27
pandas>=2.0,<2.3
torch>=2.2
transformers==4.47.1
huggingface-hub<1.0
chronos-forecasting>=1.5.0
scikit-learn>=1.4
EOF

cat > "${REQ_UNI2TS}" <<'EOF'
numpy~=1.26.0
pandas>=2.0,<2.3
torch>=2.2
transformers==4.47.1
huggingface-hub<1.0
gluonts>=0.14
EOF

cat > "${REQ_MOMENT}" <<'EOF'
numpy==1.25.2
pandas>=2.0,<2.3
torch>=2.2
transformers==4.33.3
huggingface-hub==0.24.0
momentfm==0.1.4
EOF

cat > "${REQ_LEGACY}" <<'EOF'
numpy>=1.24,<1.27
pandas>=2.0,<2.3
torch>=2.2
scikit-learn>=1.4
line_profiler>=4.1
EOF

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[setup] creating env_foundation"
create_env "env_foundation" "${PYTHON_BIN}" "${REQ_FOUNDATION}"

echo "[setup] creating env_uni2ts"
create_env "env_uni2ts" "${PYTHON_BIN}" "${REQ_UNI2TS}"
if ! "${ENV_DIR}/env_uni2ts/bin/pip" install -e "${ROOT_DIR}/external/baselines/uni2ts"; then
  echo "[setup][warn] editable uni2ts install failed; keep env for benchmark scripts only."
fi

echo "[setup] creating env_moment"
create_env "env_moment" "${PYTHON_BIN}" "${REQ_MOMENT}"

echo "[setup] creating env_legacy"
create_env "env_legacy" "${PYTHON_BIN}" "${REQ_LEGACY}"

echo "[setup] done: ${ENV_DIR}"
