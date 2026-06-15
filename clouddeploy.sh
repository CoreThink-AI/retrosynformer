#!/usr/bin/env bash
# Deploy the RetroSynFormer inference service to Google Cloud Run (GPU).
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker
#   export GCP_PROJECT=your-project-id
#
# Optional overrides:
#   export GCP_REGION=us-central1   (default)
#   export GCP_SERVICE=retrosynformer-inference  (default)
set -euo pipefail

PROJECT=${GCP_PROJECT:?Set GCP_PROJECT env var to your GCP project ID}
REGION=${GCP_REGION:-us-central1}
SERVICE=${GCP_SERVICE:-retrosynformer-inference}
SHA=$(git rev-parse --short HEAD)
IMAGE="gcr.io/${PROJECT}/${SERVICE}:${SHA}"

echo "==> Building and pushing image via Cloud Build"
gcloud builds submit \
    --tag "${IMAGE}" \
    --timeout=30m \
    --machine-type=e2-highcpu-8 \
    --project="${PROJECT}" \
    .

echo "==> Deploying to Cloud Run (GPU L4)"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=4 \
    --min-instances=1 \
    --max-instances=4 \
    --timeout=120 \
    --set-env-vars="MODEL_CONFIG_PATH=/app/model/config.yaml,MODEL_WEIGHTS_PATH=/app/model/model.pth,BUILDING_BLOCKS_PATH=/app/data/standard_building_blocks.csv,TEMPLATES_PATH=/app/data/standard_reaction_templates.pickle" \
    --set-secrets="API_KEY=retrosynformer-api-key:latest" \
    --no-allow-unauthenticated

echo "==> Service URL:"
gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --format "value(status.url)"

echo ""
echo "To test the deployed service:"
echo "  SERVICE_URL=\$(gcloud run services describe ${SERVICE} --region=${REGION} --format='value(status.url)')"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  API_KEY=\$(gcloud secrets versions access latest --secret=retrosynformer-api-key)"
echo '  curl -X POST "$SERVICE_URL/predict" \'
echo '    -H "Authorization: Bearer $TOKEN" \'
echo '    -H "X-API-Key: $API_KEY" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "beam_width": 10}'"'"
