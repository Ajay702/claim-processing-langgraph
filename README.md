# Claim Processing Pipeline

A FastAPI service that processes insurance claim PDFs through a multi-agent LangGraph workflow. Upload a PDF, and the system classifies each page by document type, extracts structured data from identity documents, discharge summaries, and itemized bills, then returns a single aggregated result.

Built with Cerebras GPT-OSS-120B for classification and extraction, PyMuPDF for PDF text extraction, and LangGraph for workflow orchestration.

## Architecture

The request flow is straightforward:

```
Client uploads PDF
        │
        ▼
   FastAPI endpoint
   (validate, save to /tmp, extract text per page)
        │
        ▼
   LangGraph Workflow
        │
        ▼
   ┌─────────────┐
   │  Segregator  │  ← classifies each page into 1 of 9 types
   └──────┬───────┘
          │
    ┌─────┼──────────┐
    ▼     ▼          ▼
┌──────┐┌──────┐┌──────┐
│  ID  ││Disch.││ Bill │  ← 3 extraction agents run in parallel
│Agent ││Agent ││Agent │
└──┬───┘└──┬───┘└──┬───┘
   │       │       │
   └───────┼───────┘
           ▼
    ┌─────────────┐
    │  Aggregator  │  ← combines all outputs + metadata
    └─────────────┘
           │
           ▼
      JSON response
```

Each agent only receives the pages relevant to it. If a PDF has no discharge summary pages, the discharge agent gets nothing and returns defaults. This keeps LLM calls focused and avoids sending irrelevant text.

## Project Structure

```
app/
├── main.py                  # FastAPI app, uvicorn entry point
├── api/
│   └── routes.py            # POST /api/process, GET /health
├── services/
│   └── pdf_parser.py        # Page-level text extraction (PyMuPDF)
└── graph/
    ├── state.py             # ClaimState TypedDict
    ├── workflow.py           # LangGraph StateGraph definition
    └── nodes/
        ├── llm_client.py    # Shared Cerebras client
        ├── segregator.py    # Page classifier
        ├── id_agent.py      # Identity extraction
        ├── discharge_agent.py # Discharge summary extraction
        ├── bill_agent.py    # Itemized bill extraction + verification
        └── aggregator.py    # Final output assembly
tests/
└── test_pipeline.py         # 13 test cases with real API calls
```

## LangGraph Workflow

The graph is defined as a `StateGraph` with a shared `ClaimState` typed dict that flows through all nodes:

```
START → segregator → id_agent      ─┐
                   → discharge_agent ├→ aggregator → END
                   → bill_agent    ─┘
```

The three extraction agents fan out from the segregator and fan back into the aggregator. LangGraph handles the parallel execution — the agents don't depend on each other, so they run concurrently.

The graph is compiled once at module level and reused for every request.

## Agents

### Segregator

Classifies each page independently into one of 9 document types:

`claim_forms`, `cheque_or_bank_details`, `identity_document`, `itemized_bill`, `discharge_summary`, `prescription`, `investigation_report`, `cash_receipt`, `other`

Each page gets its own LLM call. The output is a dict mapping document types to lists of page numbers (e.g., `{"identity_document": [2], "itemized_bill": [1, 4]}`). Empty categories are pruned.

If the LLM returns an invalid type or fails to respond, the page defaults to `other`.

### ID Agent

Receives pages classified as `identity_document`. Extracts:

- `patient_name`
- `date_of_birth`
- `policy_number`
- `member_id`
- `insurance_provider`

Fields not found in the text are returned as `null`. Confidence is computed from the ratio of non-null fields: 80%+ is `high`, 40%+ is `medium`, below that is `low`.

### Discharge Agent

Receives `discharge_summary` pages. Extracts:

- `diagnosis`
- `admission_date`
- `discharge_date`
- `treating_physician`
- `hospital_name`

Same confidence scoring as the ID agent.

### Bill Agent

Receives `itemized_bill` pages. Extracts line items with `description`, `quantity`, `unit_price`, and `total_price`.

The important part: the pipeline does not trust the LLM's math. After extraction, Python independently recalculates `total_price` for each item as `quantity * unit_price`, and sums them into a `verified_total`. If the LLM's `calculated_total` differs from the `verified_total` by more than 0.01, the response includes `total_mismatch: true`.

Missing quantities default to 1. Malformed items are silently dropped.

Confidence: 3+ items is `high`, 1-2 is `medium`, 0 is `low`.

### Aggregator

Combines all agent outputs into the final response and adds `processing_metadata`:

- `total_pages` — number of pages in the PDF
- `classified_types` — sorted list of document types found
- `timestamp` — ISO-8601 UTC timestamp

## API

### POST /api/process

Accepts a multipart form with `claim_id` (string) and `file` (PDF).

```bash
curl -X POST https://your-app.onrender.com/api/process \
  -F "claim_id=CLM-001" \
  -F "file=@claim_document.pdf"
```

Response (200):

```json
{
  "claim_id": "CLM-001",
  "filename": "claim_document.pdf",
  "pages_count": 5,
  "status": "processed",
  "output": {
    "claim_id": "CLM-001",
    "identity_info": {
      "patient_name": "Ravi Kumar",
      "date_of_birth": "15-05-1990",
      "policy_number": "POL-123",
      "member_id": "MEM-456",
      "insurance_provider": "Star Health",
      "confidence": "high"
    },
    "discharge_summary": {
      "diagnosis": "Dengue Fever",
      "admission_date": "10-Jan-2024",
      "discharge_date": "15-Jan-2024",
      "treating_physician": "Dr. Mehta",
      "hospital_name": "Apollo Hospital",
      "confidence": "high"
    },
    "billing_details": {
      "items": [
        {
          "description": "Room charges",
          "quantity": 5,
          "unit_price": 1000,
          "total_price": 5000
        }
      ],
      "calculated_total": 5000,
      "verified_total": 5000,
      "total_mismatch": false,
      "confidence": "medium"
    },
    "processing_metadata": {
      "total_pages": 5,
      "classified_types": ["discharge_summary", "identity_document", "itemized_bill"],
      "timestamp": "2024-03-15T10:30:00+00:00"
    }
  }
}
```

Error responses: `400` for invalid input, `422` for missing fields, `500` for workflow failures.

### GET /health

Returns `{"status": "ok"}`. Used by Render for health checks.

## Production Considerations

**Input validation** — The endpoint checks content type and file extension before processing. Empty claim IDs are rejected. Corrupt or image-only PDFs that yield no extractable text return a 400.

**Bill total verification** — The LLM extracts line items and reports a total, but the pipeline recalculates independently in Python. Any discrepancy is flagged with `total_mismatch: true`. This catches arithmetic errors that LLMs commonly make.

**Confidence scoring** — Each extraction agent reports a confidence level based on how many fields it managed to fill. This gives the caller a quick signal about extraction quality without needing to inspect every field.

**Error isolation** — Each agent node is wrapped in a try/except. If the ID agent fails (API timeout, bad JSON, etc.), the discharge and bill agents still run normally. The failed agent returns safe defaults with `confidence: low`. The pipeline never crashes because one agent had a bad day.

**JSON schema validation** — LLM responses are validated against expected field names and types. Non-dict responses, missing keys, and non-string values are handled with fallbacks rather than exceptions.

## Testing

The test suite (`tests/test_pipeline.py`) has 13 test cases that hit the real Cerebras API — no mocking. They cover:

1. **Mixed-order document** — Pages arrive as bill, identity, discharge. Checks correct routing.
2. **Multiple identity pages** — Two Aadhaar card pages combined for extraction.
3. **Multiple discharge pages** — Three-page discharge summary merged correctly.
4. **Incorrect printed total** — Bill with wrong total on the page. Validates mismatch detection.
5. **Missing quantities** — Bill items without explicit quantity. Checks default-to-1 behavior.
6. **Unrelated PDF** — A cooking recipe. Everything goes to `other`, all agents return defaults.
7. **Noisy/garbled text** — OCR-quality text with corrupted characters. Checks graceful degradation.
8. **Empty PDF** — Valid PDF with blank pages. Returns 400.
9. **All 9 document types** — One page per type. Verifies complete classification coverage.
10. **Partial agent failure** — ID agent is forced to fail. Other agents succeed independently.
11. **Health endpoint** — GET /health returns 200.
12. **Non-PDF rejection** — Text file upload returns 400.
13. **Missing claim_id** — Returns 422.

Run them with:

```bash
pytest tests/test_pipeline.py -v -s
```

Note: Tests 1-10 make real LLM calls, so they need `CEREBRAS_API_KEY` set and will take ~30-40 seconds total.

## Deployment

Deployed on [Render](https://render.com) as a native Python web service (no Docker).

- **Runtime**: Python 3.11
- **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment variable**: `CEREBRAS_API_KEY` must be set in Render's environment settings

The app reads `PORT` from the environment (Render sets this automatically). `python-dotenv` is included for local development but is a no-op when no `.env` file exists.

## Design Decisions

**Why separate segregation from extraction?**

A claim PDF is a stack of different documents — an Aadhaar card next to a hospital bill next to a prescription. Sending the entire PDF to one extraction prompt would force the LLM to figure out what's what and extract everything at once. That's unreliable. By classifying first, each extraction agent gets a focused context with only the pages it cares about. The prompts are simpler, the outputs are more consistent, and it's easier to debug when something goes wrong.

**Why recalculate bill totals in Python?**

LLMs are not calculators. They can extract "5 x 1000 = 5000" from text just fine, but when you ask them to sum 15 line items, they regularly get it wrong. The pipeline trusts the LLM to read the numbers off the page, then does the arithmetic itself. The `total_mismatch` flag is there so downstream systems can decide how to handle discrepancies.

**Why LangGraph instead of just chaining function calls?**

For three agents, you could get away with sequential calls. But LangGraph gives you fan-out parallelism for free — the three extraction agents run concurrently, which cuts latency. It also gives you a typed state object that flows through the graph, making it easy to add nodes later without rewiring everything. If you need to add a prescription extraction agent tomorrow, it's one node and two edges.
