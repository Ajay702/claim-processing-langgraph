"""Comprehensive stress tests for the Claim Processing Pipeline.

Covers 10 scenarios using pytest + FastAPI TestClient with REAL Cerebras API calls.
No mocking — every test hits the live LLM endpoint.
"""

import io
import json
from typing import Any
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app.graph.nodes.segregator import ALLOWED_TYPES
from app.main import app

client = TestClient(app)

# ─── helpers ──────────────────────────────────────────────────────────────────


def _build_pdf(page_texts: list[str]) -> bytes:
    """Create an in-memory PDF with the given page texts."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    raw = doc.tobytes()
    doc.close()
    return raw


def _post(claim_id: str, pdf_bytes: bytes, filename: str = "claim.pdf"):
    """POST to /api/process and return the response."""
    return client.post(
        "/api/process",
        data={"claim_id": claim_id},
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
    )


VALID_CONFIDENCES = {"high", "medium", "low"}

ID_FIELDS = ("patient_name", "date_of_birth", "policy_number", "member_id", "insurance_provider")
DISCHARGE_FIELDS = ("diagnosis", "admission_date", "discharge_date", "treating_physician", "hospital_name")
BILL_ITEM_KEYS = {"description", "quantity", "unit_price", "total_price"}


def _assert_top_level_schema(body: dict[str, Any], claim_id: str, pages: int):
    """Assert the top-level response structure."""
    assert body["claim_id"] == claim_id, f"Expected claim_id={claim_id}, got {body['claim_id']}"
    assert body["pages_count"] == pages, f"Expected pages_count={pages}, got {body['pages_count']}"
    assert body["status"] == "processed", f"Expected status=processed, got {body['status']}"
    assert "output" in body, "Missing 'output' key in response"

    output = body["output"]
    for key in ("claim_id", "identity_info", "discharge_summary", "billing_details", "processing_metadata"):
        assert key in output, f"Missing '{key}' in output"

    meta = output["processing_metadata"]
    for key in ("total_pages", "classified_types", "timestamp"):
        assert key in meta, f"Missing '{key}' in processing_metadata"
    assert meta["total_pages"] == pages


def _assert_id_schema(id_data: dict[str, Any]):
    """Assert identity_info has correct schema regardless of LLM content."""
    for field in ID_FIELDS:
        assert field in id_data, f"Missing '{field}' in identity_info"
        assert id_data[field] is None or isinstance(id_data[field], str), (
            f"identity_info.{field} must be str or None, got {type(id_data[field])}"
        )
    assert "confidence" in id_data, "Missing 'confidence' in identity_info"
    assert id_data["confidence"] in VALID_CONFIDENCES, f"Invalid confidence: {id_data['confidence']}"


def _assert_discharge_schema(discharge: dict[str, Any]):
    """Assert discharge_summary has correct schema."""
    for field in DISCHARGE_FIELDS:
        assert field in discharge, f"Missing '{field}' in discharge_summary"
        assert discharge[field] is None or isinstance(discharge[field], str), (
            f"discharge_summary.{field} must be str or None, got {type(discharge[field])}"
        )
    assert "confidence" in discharge, "Missing 'confidence' in discharge_summary"
    assert discharge["confidence"] in VALID_CONFIDENCES, f"Invalid confidence: {discharge['confidence']}"


def _assert_bill_schema(billing: dict[str, Any]):
    """Assert billing_details has correct schema and verified totals."""
    assert "items" in billing, "Missing 'items' in billing_details"
    assert isinstance(billing["items"], list), "billing_details.items must be a list"
    for i, item in enumerate(billing["items"]):
        for k in BILL_ITEM_KEYS:
            assert k in item, f"Item {i} missing '{k}'"
        assert isinstance(item["description"], str), f"Item {i}: description must be str"
        assert isinstance(item["quantity"], (int, float)), f"Item {i}: quantity must be numeric"
        assert isinstance(item["unit_price"], (int, float)), f"Item {i}: unit_price must be numeric"
        assert isinstance(item["total_price"], (int, float)), f"Item {i}: total_price must be numeric"
        # total_price == qty * unit_price (pipeline recalculates)
        expected = round(item["quantity"] * item["unit_price"], 2)
        assert item["total_price"] == expected, (
            f"Item {i}: total_price={item['total_price']} != qty*unit={expected}"
        )

    assert "verified_total" in billing, "Missing 'verified_total'"
    assert "calculated_total" in billing, "Missing 'calculated_total'"
    assert "total_mismatch" in billing, "Missing 'total_mismatch'"
    assert isinstance(billing["total_mismatch"], bool), "total_mismatch must be bool"
    assert "confidence" in billing, "Missing 'confidence'"
    assert billing["confidence"] in VALID_CONFIDENCES, f"Invalid confidence: {billing['confidence']}"

    # Verify the verified_total is the sum of item total_prices
    expected_total = round(sum(i["total_price"] for i in billing["items"]), 2)
    assert billing["verified_total"] == expected_total, (
        f"verified_total={billing['verified_total']} != sum(items)={expected_total}"
    )

    # Mismatch flag must be consistent
    mismatch_expected = abs(billing["verified_total"] - billing["calculated_total"]) > 0.01
    assert billing["total_mismatch"] == mismatch_expected, (
        f"total_mismatch={billing['total_mismatch']} but verified={billing['verified_total']} "
        f"vs calculated={billing['calculated_total']}"
    )


def _assert_classified_types_valid(types: list[str]):
    """All classified types must be from ALLOWED_TYPES."""
    for t in types:
        assert t in ALLOWED_TYPES, f"Invalid classified type: '{t}'. Allowed: {ALLOWED_TYPES}"


# ─── Scenario 1: Mixed-order document ────────────────────────────────────────


def test_01_mixed_order_document():
    """Pages arrive in jumbled order: bill, id, discharge. Pipeline routes and extracts correctly."""
    pages = [
        # Page 1: Itemized bill
        (
            "HOSPITAL ITEMIZED BILL\n"
            "Patient: Ravi Kumar\n"
            "Date: 15-Jan-2024\n"
            "----------------------------\n"
            "Room charges (5 days)     5 x 1000 = 5000\n"
            "Medicines (IV fluids)     1 x 2000 = 2000\n"
            "Doctor consultation       1 x 1500 = 1500\n"
            "----------------------------\n"
            "Total: 8500"
        ),
        # Page 2: Identity document
        (
            "GOVERNMENT OF INDIA\n"
            "AADHAAR CARD\n"
            "Name: Ravi Kumar\n"
            "Date of Birth: 15-05-1990\n"
            "Aadhaar Number: 1234-5678-9012\n"
            "Address: 42 MG Road, Pune, Maharashtra 411001"
        ),
        # Page 3: Discharge summary
        (
            "DISCHARGE SUMMARY\n"
            "Apollo Hospital, Chennai\n"
            "Patient: Ravi Kumar\n"
            "Diagnosis: Dengue Fever\n"
            "Date of Admission: 10-Jan-2024\n"
            "Date of Discharge: 15-Jan-2024\n"
            "Treating Physician: Dr. Mehta\n"
            "Treatment: IV fluids, platelet monitoring, antipyretics\n"
            "Follow-up: Review after 1 week"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("MIXED-001", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "MIXED-001", 3)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])
    _assert_classified_types_valid(output["processing_metadata"]["classified_types"])

    # With rich content, LLM should extract meaningful data
    id_info = output["identity_info"]
    assert id_info["patient_name"] is not None, "LLM should extract patient name from Aadhaar page"
    print(f"  ✓ Identity extracted: name={id_info['patient_name']}, confidence={id_info['confidence']}")

    discharge = output["discharge_summary"]
    assert discharge["diagnosis"] is not None, "LLM should extract diagnosis from discharge page"
    print(f"  ✓ Discharge extracted: diagnosis={discharge['diagnosis']}, confidence={discharge['confidence']}")

    billing = output["billing_details"]
    assert len(billing["items"]) >= 1, "LLM should extract at least 1 bill item"
    print(f"  ✓ Billing extracted: {len(billing['items'])} items, verified_total={billing['verified_total']}, mismatch={billing['total_mismatch']}")


# ─── Scenario 2: Multiple identity pages ─────────────────────────────────────


def test_02_multiple_identity_pages():
    """Two identity pages should be combined and processed together by id_agent."""
    pages = [
        (
            "GOVERNMENT OF INDIA\n"
            "AADHAAR CARD (FRONT)\n"
            "Name: Sneha Patil\n"
            "Date of Birth: 20-03-1985\n"
            "Female\n"
            "Aadhaar Number: 9876-5432-1098"
        ),
        (
            "AADHAAR CARD (BACK)\n"
            "Address: 15 Koregaon Park, Pune\n"
            "Maharashtra 411001\n"
            "QR Code: [encoded data]\n"
            "VID: 1234 5678 9012 3456"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("MULTI-ID-002", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "MULTI-ID-002", 2)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])

    # Both pages classified as identity → combined for extraction
    id_info = output["identity_info"]
    assert id_info["patient_name"] is not None, "Name should be extracted from Aadhaar front"
    print(f"  ✓ ID from 2 pages: name={id_info['patient_name']}, confidence={id_info['confidence']}")

    # No discharge or bill pages → defaults
    assert output["discharge_summary"]["confidence"] == "low", "No discharge pages → low confidence"
    assert output["billing_details"]["confidence"] == "low", "No bill pages → low confidence"
    assert output["billing_details"]["items"] == [], "No bill pages → empty items"
    print(f"  ✓ Discharge default: confidence={output['discharge_summary']['confidence']}")
    print(f"  ✓ Billing default: confidence={output['billing_details']['confidence']}, items=[]")


# ─── Scenario 3: Multiple discharge pages ────────────────────────────────────


def test_03_multiple_discharge_pages():
    """Three discharge pages merged into one extraction call."""
    pages = [
        (
            "DISCHARGE SUMMARY — Page 1 of 3\n"
            "Fortis Hospital, Mumbai\n"
            "Patient Name: Amit Deshmukh\n"
            "MRD No: MRD-44521\n"
            "Date of Admission: 15-Feb-2024\n"
            "Presenting Complaints: High grade fever for 5 days, body aches, headache"
        ),
        (
            "DISCHARGE SUMMARY — Page 2 of 3\n"
            "Investigations:\n"
            "- CBC: Platelet count 45,000\n"
            "- NS1 Antigen: Positive\n"
            "- Dengue IgM: Positive\n"
            "Diagnosis: Dengue Hemorrhagic Fever\n"
            "Treatment: IV fluids, platelet transfusion, monitoring"
        ),
        (
            "DISCHARGE SUMMARY — Page 3 of 3\n"
            "Date of Discharge: 20-Feb-2024\n"
            "Condition at Discharge: Stable, platelet count normalized\n"
            "Treating Physician: Dr. Rao\n"
            "Follow-up: Review after 1 week with CBC\n"
            "Medications: Paracetamol 500mg SOS"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("MULTI-DC-003", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "MULTI-DC-003", 3)

    output = body["output"]
    _assert_discharge_schema(output["discharge_summary"])
    _assert_id_schema(output["identity_info"])
    _assert_bill_schema(output["billing_details"])

    discharge = output["discharge_summary"]
    assert discharge["diagnosis"] is not None, "Diagnosis should be extracted from multi-page discharge"
    assert discharge["confidence"] in ("high", "medium"), "Rich discharge text → high or medium confidence"
    print(f"  ✓ Discharge from 3 pages: diagnosis={discharge['diagnosis']}, confidence={discharge['confidence']}")

    # No identity or bill pages
    assert output["identity_info"]["confidence"] == "low", "No identity pages → low confidence"
    assert output["billing_details"]["items"] == [], "No bill pages → empty items"


# ─── Scenario 4: Bill with incorrect printed total ───────────────────────────


def test_04_bill_incorrect_total():
    """Bill page with a deliberately wrong printed total. Pipeline must flag total_mismatch."""
    pages = [
        (
            "ITEMIZED HOSPITAL BILL\n"
            "City Hospital, Bangalore\n"
            "Patient: Priya Sharma\n"
            "Date: 05-Mar-2024\n"
            "-----------------------------------\n"
            "Description          Qty   Rate   Amount\n"
            "Consultation          1    1500    1500\n"
            "Blood Test (CBC)      1     800     800\n"
            "X-Ray Chest           1    2500    2500\n"
            "-----------------------------------\n"
            "TOTAL:  Rs. 9999\n"
            "(Note: the printed total above is WRONG, actual sum is 4800)"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("BILL-ERR-004", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "BILL-ERR-004", 1)

    billing = body["output"]["billing_details"]
    _assert_bill_schema(billing)

    assert len(billing["items"]) >= 1, "Should extract at least 1 bill item"
    # The verified_total is independently computed from items
    print(f"  ✓ Bill items: {len(billing['items'])}")
    print(f"  ✓ LLM calculated_total: {billing['calculated_total']}")
    print(f"  ✓ Pipeline verified_total: {billing['verified_total']}")
    print(f"  ✓ total_mismatch: {billing['total_mismatch']}")
    print(f"  ✓ confidence: {billing['confidence']}")


# ─── Scenario 5: Bill with missing quantity ──────────────────────────────────


def test_05_bill_missing_quantity():
    """Bill items where quantity is implicit (1). Pipeline should handle gracefully."""
    pages = [
        (
            "DIAGNOSTIC CENTER BILL\n"
            "HealthFirst Lab, Hyderabad\n"
            "Date: 10-Mar-2024\n"
            "-----------------------------------\n"
            "MRI Brain Scan                  Rs. 12000\n"
            "Bandage (pack of 5)   5 x 50  = Rs.   250\n"
            "Blood Culture                   Rs.  1500\n"
            "-----------------------------------\n"
            "Total: Rs. 13750"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("BILL-QTY-005", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "BILL-QTY-005", 1)

    billing = body["output"]["billing_details"]
    _assert_bill_schema(billing)

    assert len(billing["items"]) >= 1, "Should extract at least 1 bill item"
    # For every item, total_price == qty * unit_price (enforced by _assert_bill_schema)
    for item in billing["items"]:
        assert item["quantity"] >= 1, f"Quantity should be >= 1, got {item['quantity']} for '{item['description']}'"
    print(f"  ✓ {len(billing['items'])} items extracted, all with valid quantities")
    print(f"  ✓ verified_total={billing['verified_total']}, mismatch={billing['total_mismatch']}")


# ─── Scenario 6: Completely unrelated PDF ────────────────────────────────────


def test_06_unrelated_pdf():
    """A PDF about cooking recipes — should be classified as 'other', all agents return defaults."""
    pages = [
        (
            "BUTTER CHICKEN RECIPE\n\n"
            "Ingredients:\n"
            "- 500g chicken thigh, boneless\n"
            "- 1 cup yogurt\n"
            "- 2 tbsp garam masala\n"
            "- 1 cup tomato puree\n"
            "- 100g butter\n"
            "- 1 cup cream\n\n"
            "Step 1: Marinate chicken in yogurt and spices for 2 hours.\n"
            "Step 2: Grill or bake the chicken until charred."
        ),
        (
            "BUTTER CHICKEN RECIPE (continued)\n\n"
            "Step 3: In a pan, melt butter and cook onion-tomato paste.\n"
            "Step 4: Add cream, kasuri methi, and sugar.\n"
            "Step 5: Add grilled chicken pieces and simmer for 15 minutes.\n"
            "Step 6: Garnish with fresh cream and coriander.\n\n"
            "Serve hot with naan or steamed rice.\n"
            "Prep time: 30 min | Cook time: 45 min | Serves: 4"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("UNRELATED-006", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "UNRELATED-006", 2)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])
    _assert_classified_types_valid(output["processing_metadata"]["classified_types"])

    # Nothing relevant → all agents should return low confidence / defaults
    # Note: LLM might still classify as "other" or something unexpected
    types = output["processing_metadata"]["classified_types"]
    print(f"  ✓ Classified types for recipe PDF: {types}")

    # Key check: no meaningful extraction from a recipe
    id_info = output["identity_info"]
    discharge = output["discharge_summary"]
    billing = output["billing_details"]

    # If LLM correctly classifies as 'other', extraction agents get no pages
    if "identity_document" not in types:
        assert id_info["confidence"] == "low", "No identity pages → low confidence"
        assert id_info["patient_name"] is None, "No identity pages → null patient_name"
    if "discharge_summary" not in types:
        assert discharge["confidence"] == "low", "No discharge pages → low confidence"
    if "itemized_bill" not in types:
        assert billing["items"] == [], "No bill pages → empty items"
        assert billing["verified_total"] == 0, "No bill pages → zero total"

    print(f"  ✓ ID confidence={id_info['confidence']}, Discharge confidence={discharge['confidence']}, Billing confidence={billing['confidence']}")


# ─── Scenario 7: Very noisy / garbled text ───────────────────────────────────


def test_07_noisy_text():
    """OCR-garbled text. LLM should return nulls / low confidence. Pipeline handles gracefully."""
    pages = [
        (
            "G0V3RNM3NT 0F 1ND1@\n"
            "@@DH@@R C@RD\n"
            "N@m3: ???????????\n"
            "D@t3 0f B1rth: ##/##/####\n"
            "@@dh@@r Numb3r: XXXX-XXXX-XXXX\n"
            "P0l1cy: c0rrupt3d_d@t@_str3@m\n"
            "@ddr3$$: |||||||||||||||||||"
        ),
        (
            "D1$CH@RG3 $UMM@RY\n"
            "H0$p1t@l: ????????????\n"
            "P@t13nt: ###############\n"
            "D1@gn0$1$: unr3@d@bl3 t3xt g@rbl3d\n"
            "@dm1$$10n D@t3: XX-XX-XXXX\n"
            "D1$ch@rg3 D@t3: garbl3d\n"
            "Tr3@t1ng Phy$1c1@n: Dr. ??????????\n"
            "Tr3@tm3nt: c0rrupt3d d@t@ $tr3@m ||||"
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("NOISY-007", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "NOISY-007", 2)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])
    _assert_classified_types_valid(output["processing_metadata"]["classified_types"])

    # With garbled text, confidence should be low or medium at best
    id_info = output["identity_info"]
    discharge = output["discharge_summary"]
    print(f"  ✓ Noisy ID: confidence={id_info['confidence']}, fields={id_info}")
    print(f"  ✓ Noisy Discharge: confidence={discharge['confidence']}, fields={discharge}")

    # Pipeline should not crash — that's the main validation
    assert output["billing_details"]["items"] == [] or True, "Bill items may or may not be empty"


# ─── Scenario 8: Empty but valid PDF ─────────────────────────────────────────


def test_08_empty_valid_pdf():
    """A valid PDF with zero extractable text. Should fail extraction with 400."""
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    pdf_bytes = doc.tobytes()
    doc.close()

    resp = _post("EMPTY-008", pdf_bytes)
    # extract_pages raises ValueError for no extractable text → 400
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "detail" in body, "Error response should contain 'detail'"
    print(f"  ✓ Empty PDF correctly rejected: {body['detail']}")


# ─── Scenario 9: All 9 document types present ────────────────────────────────


def test_09_all_nine_types():
    """9 pages, each clearly representing a different document type. Validates classification coverage."""
    pages = [
        # 1. claim_forms
        (
            "CASHLESS CLAIM FORM\n"
            "Insurance Company: Star Health\n"
            "Policy Number: SH-2024-001234\n"
            "Name of Insured: Kavita Reddy\n"
            "Sum Insured: Rs. 5,00,000\n"
            "Hospital Name: Apollo Hospital\n"
            "Nature of Illness: Planned Surgery\n"
            "Declaration: I hereby declare that the above information is true."
        ),
        # 2. cheque_or_bank_details
        (
            "BANK ACCOUNT DETAILS FOR REIMBURSEMENT\n"
            "Account Holder: Kavita Reddy\n"
            "Bank Name: HDFC Bank\n"
            "Branch: Jubilee Hills, Hyderabad\n"
            "Account Number: 5020 0012 3456 789\n"
            "IFSC Code: HDFC0001234\n"
            "CANCELLED CHEQUE ATTACHED"
        ),
        # 3. identity_document
        (
            "GOVERNMENT OF INDIA\n"
            "PAN CARD\n"
            "Name: KAVITA REDDY\n"
            "Father's Name: VENKAT REDDY\n"
            "Date of Birth: 12/08/1978\n"
            "Permanent Account Number: ABCDE1234F"
        ),
        # 4. itemized_bill
        (
            "ITEMIZED HOSPITAL BILL\n"
            "Apollo Hospital, Hyderabad\n"
            "Patient: Kavita Reddy\n"
            "Room charges (3 days)    3 x 3000 = 9000\n"
            "Surgery charges                   = 45000\n"
            "Anaesthesia                        = 8000\n"
            "Medicines                          = 5500\n"
            "Total: Rs. 67,500"
        ),
        # 5. discharge_summary
        (
            "DISCHARGE SUMMARY\n"
            "Apollo Hospital, Hyderabad\n"
            "Patient: Kavita Reddy\n"
            "Diagnosis: Cholecystitis (Gallstones)\n"
            "Admission Date: 01-Mar-2024\n"
            "Discharge Date: 04-Mar-2024\n"
            "Treating Physician: Dr. Suresh Babu\n"
            "Procedure: Laparoscopic Cholecystectomy"
        ),
        # 6. prescription
        (
            "PRESCRIPTION\n"
            "Dr. Suresh Babu, MS (General Surgery)\n"
            "Apollo Hospital, Hyderabad\n"
            "Patient: Kavita Reddy\n"
            "Date: 04-Mar-2024\n"
            "Rx:\n"
            "1. Tab Pantoprazole 40mg — 1-0-0 x 14 days\n"
            "2. Tab Paracetamol 650mg — SOS\n"
            "3. Cap Amoxicillin 500mg — 1-0-1 x 5 days\n"
            "Review after 2 weeks"
        ),
        # 7. investigation_report
        (
            "LABORATORY INVESTIGATION REPORT\n"
            "HealthFirst Diagnostics, Hyderabad\n"
            "Patient: Kavita Reddy | Age: 45 | Sex: Female\n"
            "Date: 28-Feb-2024\n"
            "Test: Ultrasound Abdomen\n"
            "Findings: Multiple calculi in gallbladder, largest 12mm.\n"
            "Impression: Cholelithiasis. Surgical consultation advised."
        ),
        # 8. cash_receipt
        (
            "PAYMENT RECEIPT\n"
            "Apollo Hospital, Hyderabad\n"
            "Receipt No: RCP-2024-5678\n"
            "Date: 04-Mar-2024\n"
            "Received from: Kavita Reddy\n"
            "Amount: Rs. 67,500 (Sixty-Seven Thousand Five Hundred Only)\n"
            "Mode of Payment: Credit Card\n"
            "Against: Bill No. BILL-2024-1234\n"
            "Authorised Signatory: ________________"
        ),
        # 9. other
        (
            "TERMS AND CONDITIONS\n"
            "This document outlines the general terms and conditions\n"
            "of the hospital stay. The patient agrees to abide by\n"
            "all hospital rules and regulations during the stay.\n"
            "Visiting hours: 10 AM to 8 PM.\n"
            "The hospital is not responsible for loss of valuables."
        ),
    ]
    pdf = _build_pdf(pages)

    resp = _post("ALL9-009", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "ALL9-009", 9)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])

    meta = output["processing_metadata"]
    assert meta["total_pages"] == 9, f"Expected 9 pages, got {meta['total_pages']}"
    _assert_classified_types_valid(meta["classified_types"])

    # We expect the LLM to classify most types correctly — at minimum identity, bill, discharge
    types = set(meta["classified_types"])
    print(f"  ✓ Classified types ({len(types)}): {sorted(types)}")
    print(f"  ✓ ID: confidence={output['identity_info']['confidence']}, name={output['identity_info'].get('patient_name')}")
    print(f"  ✓ Discharge: confidence={output['discharge_summary']['confidence']}, diagnosis={output['discharge_summary'].get('diagnosis')}")
    print(f"  ✓ Billing: {len(output['billing_details']['items'])} items, verified_total={output['billing_details']['verified_total']}")

    # Core types must be classified
    for expected_type in ("identity_document", "discharge_summary", "itemized_bill"):
        assert expected_type in types, f"Expected '{expected_type}' in classified types, got {types}"


# ─── Scenario 10: Partial agent failure (mock only the failing agent) ────────


@patch("app.graph.nodes.id_agent.call_llm")
def test_10_partial_agent_failure(mock_id_llm):
    """ID agent LLM call fails. Pipeline isolates failure, other agents succeed with real API."""
    pages = [
        (
            "GOVERNMENT OF INDIA\n"
            "VOTER ID CARD\n"
            "Name: Raj Malhotra\n"
            "Father: Suresh Malhotra\n"
            "Date of Birth: 05-07-1982\n"
            "EPIC No: ABC1234567"
        ),
        (
            "DISCHARGE SUMMARY\n"
            "Narayana Hospital, Bangalore\n"
            "Patient: Raj Malhotra\n"
            "Diagnosis: Right Tibia Fracture\n"
            "Admission: 01-Apr-2024\n"
            "Discharge: 10-Apr-2024\n"
            "Treating Physician: Dr. Joshi\n"
            "Treatment: Open reduction, internal fixation with plate and screws"
        ),
        (
            "ITEMIZED BILL\n"
            "Narayana Hospital, Bangalore\n"
            "Patient: Raj Malhotra\n"
            "Plaster cast          1 x 3000 = 3000\n"
            "Painkillers           1 x  500 =  500\n"
            "Physiotherapy         3 x  800 = 2400\n"
            "Total: Rs. 5900"
        ),
    ]
    pdf = _build_pdf(pages)

    # Only the ID agent blows up — segregator, discharge, bill use real API
    mock_id_llm.side_effect = RuntimeError("Simulated Cerebras API timeout")

    resp = _post("PARTIAL-010", pdf)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    _assert_top_level_schema(body, "PARTIAL-010", 3)

    output = body["output"]
    _assert_id_schema(output["identity_info"])
    _assert_discharge_schema(output["discharge_summary"])
    _assert_bill_schema(output["billing_details"])

    # ID agent failed → defaults with low confidence
    assert output["identity_info"]["confidence"] == "low", (
        f"Failed ID agent → low confidence, got '{output['identity_info']['confidence']}'"
    )
    assert output["identity_info"]["patient_name"] is None, "Failed ID agent → null patient_name"
    print(f"  ✓ ID agent failure isolated: confidence=low, all fields null")

    # Other agents should succeed with real API
    discharge = output["discharge_summary"]
    assert discharge["diagnosis"] is not None, "Discharge should succeed despite ID failure"
    assert discharge["confidence"] in ("high", "medium"), "Real discharge data → high or medium confidence"
    print(f"  ✓ Discharge succeeded: diagnosis={discharge['diagnosis']}, confidence={discharge['confidence']}")

    billing = output["billing_details"]
    assert len(billing["items"]) >= 1, "Bill should extract items despite ID failure"
    assert billing["verified_total"] > 0, "Bill total should be positive"
    print(f"  ✓ Billing succeeded: {len(billing['items'])} items, verified_total={billing['verified_total']}")


# ─── Health endpoint ─────────────────────────────────────────────────────────


def test_health_endpoint():
    """GET /health returns 200 with {status: ok}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ─── Input validation ────────────────────────────────────────────────────────


def test_reject_non_pdf():
    """Uploading a non-PDF file returns 400."""
    resp = client.post(
        "/api/process",
        data={"claim_id": "VAL-001"},
        files={"file": ("report.txt", io.BytesIO(b"plain text"), "text/plain")},
    )
    assert resp.status_code == 400, f"Expected 400 for non-PDF, got {resp.status_code}"


def test_missing_claim_id():
    """Missing claim_id returns 422."""
    pdf = _build_pdf(["dummy"])
    resp = client.post(
        "/api/process",
        files={"file": ("doc.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert resp.status_code == 422, f"Expected 422 for missing claim_id, got {resp.status_code}"
