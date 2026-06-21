# RetroSynFormer Inference Endpoint

**Service:** `retrosynformer-inference-v3`  
**Base URL:** `https://retrosynformer-inference-v3-125069248164.us-central1.run.app`  
**Interactive docs:** `/docs` (Swagger UI — requires auth header added manually)

---

## Authentication

Every request requires two headers:

```bash
Authorization: Bearer $(gcloud auth print-identity-token)
X-API-Key: <api-key>
```

Retrieve the API key once:

```bash
export RETRO_API_KEY=$(gcloud secrets versions access latest \
  --secret=retrosynformer-api-key \
  --project=biochem-db-by-hobs)
```

Tokens from `gcloud auth print-identity-token` expire after ~1 hour; regenerate as needed.

---

## Endpoints

### `GET /health`

Returns model load status, device, and default beam width.

```bash
curl -s \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "X-API-Key: $RETRO_API_KEY" \
  https://retrosynformer-inference-v3-125069248164.us-central1.run.app/health | jq .
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "beam_width_default": 10,
  "action_dim": 2957
}
```

---

### `POST /retrosynthesis`  ← primary endpoint

Returns retrosynthesis routes in the synthesis-routes-generator wire format.
Routes appear in `ai_routes`; `literature_routes` is always empty.

#### Request

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `smiles` | string | required | — | Target molecule (canonical or isomeric SMILES) |
| `max_steps` | int | 6 | 1–20 | Maximum disconnection depth (= max reaction steps per route) |
| `max_routes` | int | 5 | 1–50 | Maximum number of routes to return |
| `molecule_name` | string | null | — | Optional label; not used by the model |

#### Response

```json
{
  "target_smiles": "<input SMILES>",
  "canonical_smiles": "<RDKit canonical form>",
  "literature_routes": [],
  "ai_routes": [
    {
      "model": "model1",
      "score": 0.0173,
      "depth": 2,
      "all_leaves_purchasable": true,
      "leaf_molecules": [
        {"smiles": "Cc1ccc(Cl)nc1", "purchasable": true},
        {"smiles": "CS(=O)(=O)c1ccc(B(O)O)cc1", "purchasable": true}
      ],
      "steps": [
        {
          "step": 1,
          "target": "Cc1ccc(-c2ncc(Cl)cc2-c2ccc(S(C)(=O)=O)cc2)cn1",
          "reaction_id": "template_idx_1042",
          "reactants": ["Cc1ccc(Cl)nc1", "CS(=O)(=O)c1ccc(B(O)O)cc1"],
          "reagents": [],
          "co_products": [],
          "yield_pct": null,
          "confidence": 0.0173,
          "source": "retrosynformer",
          "dataset_name": "paroutes",
          "doi": "10.26434/chemrxiv-2025-kd6gb"
        }
      ]
    }
  ]
}
```

`all_leaves_purchasable: true` means a complete route to purchasable building blocks was found. Routes are sorted by score (highest first).

#### Example — Aspirin (simple molecule, defaults are fine)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "X-API-Key: $RETRO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}' \
  https://retrosynformer-inference-v3-125069248164.us-central1.run.app/retrosynthesis | jq '.ai_routes[0]'
```

#### Example — complex molecule (increase depth and routes)

For molecules with complexity > 600 or known multi-step syntheses, use `max_steps=10`
and `max_routes=10` (or higher). The default `max_steps=6` cuts off deep routes before
they can reach purchasable building blocks.

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "X-API-Key: $RETRO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(C)(C)c1ccc(-c2nc3ccc(C(=O)Nc4cccc(NC5=O)c4)cc3s2)cc1",
    "max_steps": 10,
    "max_routes": 10,
    "molecule_name": "Imatinib"
  }' \
  https://retrosynformer-inference-v3-125069248164.us-central1.run.app/retrosynthesis | jq '{solved: [.ai_routes[] | select(.all_leaves_purchasable)] | length, total: (.ai_routes | length)}'
```

**Guidance for `max_steps` and `max_routes`:**

| Molecule complexity (Bertz) | Recommended `max_steps` | Recommended `max_routes` | Typical time |
|-----------------------------|------------------------|--------------------------|-------------|
| < 300 (simple drug-like)    | 6 (default) | 5 (default) | ~40 s |
| 300–700 (medium)            | 8–10 | 10 | ~45 s |
| > 700 (complex / natural product) | 10–15 | 10 | ~50–60 s |

Increasing `max_routes` runs more parallel beam-search branches; increasing `max_steps`
allows deeper disconnection trees. Both increase cost roughly linearly; the endpoint
timeout is 600 s.

---

### `POST /predict`  ← raw beam-search output

Returns all beam-search routes in the model's native schema, including unsolved routes
and per-step reaction SMARTS. Useful for debugging or downstream analysis.

#### Request

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `smiles` | string | required | — | Target molecule SMILES |
| `beam_width` | int | 10 | 1–50 | Number of parallel beams (≈ max routes returned) |
| `max_depth` | int | null | 1–20 | Max depth; null uses the model config value (typically 6) |
| `target_reward` | float | 0.5 | 0–1 | Reward threshold for marking a route "solved" |
| `sort_on` | string | `"trajectory_prob"` | `"trajectory_prob"` \| `"total_reward"` | Route ranking key |

#### Response

```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "n_routes": 1,
  "n_solved": 1,
  "elapsed_s": 62.5,
  "routes": [
    {
      "route_solved": true,
      "trajectory_prob": 1.0,
      "n_steps": 0,
      "reactions": [],
      "leaf_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
      "dead_ends": []
    }
  ]
}
```

`route_solved: true` means all leaf SMILES are purchasable building blocks.
`trajectory_prob` is the product of per-step template probabilities.

---

## Python client example

```python
import json, subprocess, urllib.request

BASE = "https://retrosynformer-inference-v3-125069248164.us-central1.run.app"

def get_headers():
    token = subprocess.check_output(["gcloud", "auth", "print-identity-token"]).decode().strip()
    api_key = subprocess.check_output([
        "gcloud", "secrets", "versions", "access", "latest",
        "--secret=retrosynformer-api-key", "--project=biochem-db-by-hobs",
    ]).decode().strip()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-API-Key": api_key,
    }

def predict_routes(smiles: str, max_steps: int = 6, max_routes: int = 5) -> dict:
    payload = json.dumps({
        "smiles": smiles,
        "max_steps": max_steps,
        "max_routes": max_routes,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/retrosynthesis",
        data=payload,
        headers=get_headers(),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())

# Simple molecule — defaults fine
result = predict_routes("CC(=O)Oc1ccccc1C(=O)O")

# Complex molecule — increase depth and route count
result = predict_routes(
    "Cc1nc2ccc(NC(=O)c3ccc(CN4CCN(C)CC4)cc3)cc2n1Cc1cccnc1",
    max_steps=10,
    max_routes=10,
)

solved = [r for r in result["ai_routes"] if r["all_leaves_purchasable"]]
print(f"Solved {len(solved)}/{len(result['ai_routes'])} routes")
for r in solved:
    print(f"  depth={r['depth']}  score={r['score']:.4f}  leaves={[l['smiles'] for l in r['leaf_molecules']]}")
```

---

## Notes

- **SMILES validation**: RDKit canonicalises the input SMILES. Invalid SMILES returns HTTP 422.
  If a molecule fails, try the isomeric SMILES (`smiles` field in PubChem) rather than
  the canonical form — some stereocentre encodings are rejected by older RDKit versions.
- **Unsolved routes**: when no route has `all_leaves_purchasable: true`, `ai_routes` is
  still populated with the best partial routes; the beam exhausted at `max_steps`.
  Increasing `max_steps` or `max_routes` may help.
- **Concurrency**: the service handles one inference call at a time (semaphore-gated).
  Concurrent requests queue; do not set timeouts shorter than 300 s.
- **Model**: hypertune-large-emma-24-26_layer / trial_002, epoch 59.
  Template vocabulary: 2957 templates (large PaRoutes dataset).
