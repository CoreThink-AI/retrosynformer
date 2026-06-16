#!/usr/bin/env bash
# Deploy the RetroSynFormer inference service to Google Cloud Run (GPU).
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker
#   export GCP_PROJECT=your-project-id
#
# Optional overrides:
#   export GCP_REGION=us-central1              (default)
#   export GCP_SERVICE=retrosynformer-inference (default)
#   export GCS_BUCKET=biochem-db-by-hobs       (default)
#   export MODEL_TRIAL=trial_000               (default — best model)
set -euo pipefail

PROJECT=${GCP_PROJECT:?Set GCP_PROJECT env var to your GCP project ID}
REGION=${GCP_REGION:-us-central1}
SERVICE=${GCP_SERVICE:-retrosynformer-inference}
GCS_BUCKET=${GCS_BUCKET:-biochem-db-by-hobs}
MODEL_TRIAL=${MODEL_TRIAL:-trial_000}
SHA=$(git rev-parse --short HEAD)
IMAGE="gcr.io/${PROJECT}/${SERVICE}:${SHA}"

# GCS URIs for model artifacts (downloaded at container startup, not baked into image)
MODEL_WEIGHTS_GCS="gs://${GCS_BUCKET}/retrosynformer/models/${MODEL_TRIAL}/model.pth"
MODEL_CONFIG_GCS="gs://${GCS_BUCKET}/retrosynformer/models/${MODEL_TRIAL}/config.yaml"
BUILDING_BLOCKS_GCS="gs://${GCS_BUCKET}/retrosynformer/data/small_building_blocks.csv"
TEMPLATES_GCS="gs://${GCS_BUCKET}/retrosynformer/data/small_reaction_templates.pickle"

echo "==> Granting Cloud Run service account read access to gs://${GCS_BUCKET}"
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')
CR_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="serviceAccount:${CR_SA}" \
    --role="roles/storage.objectViewer" \
    --project="${PROJECT}"

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
    --execution-environment gen2 \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=4 \
    --min-instances=1 \
    --max-instances=4 \
    --timeout=120 \
    --set-env-vars="MODEL_WEIGHTS_GCS=${MODEL_WEIGHTS_GCS},MODEL_CONFIG_GCS=${MODEL_CONFIG_GCS},BUILDING_BLOCKS_GCS=${BUILDING_BLOCKS_GCS},TEMPLATES_GCS=${TEMPLATES_GCS},MODEL_WEIGHTS_PATH=/tmp/model/model.pth,MODEL_CONFIG_PATH=/tmp/model/config.yaml,BUILDING_BLOCKS_PATH=/tmp/data/small_building_blocks.csv,TEMPLATES_PATH=/tmp/data/small_reaction_templates.pickle" \
    --set-secrets="API_KEY=retrosynformer-api-key:latest" \
    --no-allow-unauthenticated

echo "==> Service URL:"
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --format "value(status.url)")
echo "${SERVICE_URL}"

echo ""
echo "To test the deployed service:"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  API_KEY=\$(gcloud secrets versions access latest --secret=retrosynformer-api-key --project=${PROJECT})"
echo "  curl -X POST \"${SERVICE_URL}/health\""
echo "  curl -X POST \"${SERVICE_URL}/predict\" \\"
echo "    -H \"Authorization: Bearer \$TOKEN\" \\"
echo "    -H \"X-API-Key: \$API_KEY\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"smiles\": \"CC(=O)Oc1ccccc1C(=O)O\", \"beam_width\": 10}'"
