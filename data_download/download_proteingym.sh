set -euo pipefail

VERSION="v1.3"
BASE_URL="https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}"

FILES=(
  "DMS_ProteinGym_substitutions.zip"
  "zero_shot_substitutions_scores.zip"
  "DMS_msa_files.zip"
)

mkdir -p zips extracted logs

for FILENAME in "${FILES[@]}"; do
    echo "========================================"
    echo "Downloading ${FILENAME}"
    echo "========================================"

    curl -k -L -C - \
        --retry 10 \
        --retry-delay 20 \
        --connect-timeout 60 \
        -o "zips/${FILENAME}" \
        "${BASE_URL}/${FILENAME}"

    echo "Checking zip: ${FILENAME}"
    unzip -t "zips/${FILENAME}" | tee "logs/${FILENAME}.check.log"

    echo "Extracting ${FILENAME}"
    unzip -n "zips/${FILENAME}" -d extracted/

    echo "Done: ${FILENAME}"
done

echo "All ProteinGym files downloaded and extracted."
