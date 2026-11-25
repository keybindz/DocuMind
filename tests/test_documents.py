import io

from docx import Document as DocxDocument


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_upload_txt_extracts_and_marks_ready(client, auth_headers):
    content = b"Hello DocuMind. This is a test document about backend architecture."
    resp = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert body["char_count"] == len(content)
    assert body["filename"] == "notes.txt"

    content_resp = client.get(f"/v1/documents/{body['id']}/content", headers=auth_headers)
    assert content_resp.status_code == 200
    assert "backend architecture" in content_resp.json()["extracted_text"]


def test_upload_docx_extracts_text(client, auth_headers):
    docx_bytes = _make_docx_bytes(["First paragraph.", "Second paragraph about vector databases."])
    resp = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={
            "file": (
                "report.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "ready"

    content_resp = client.get(f"/v1/documents/{body['id']}/content", headers=auth_headers)
    assert "vector databases" in content_resp.json()["extracted_text"]


def test_upload_corrupt_pdf_is_accepted_then_marked_failed(client, auth_headers):
    # Valid mime type, garbage bytes -> extraction fails gracefully rather than 500ing.
    resp = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("fake.pdf", b"this is not a real pdf", "application/pdf")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error_message"]


def test_upload_rejects_unsupported_mime_type(client, auth_headers):
    resp = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("image.png", b"\x89PNG...", "image/png")},
    )
    assert resp.status_code == 415


def test_upload_rejects_empty_file(client, auth_headers):
    resp = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400


def test_duplicate_upload_returns_existing_document(client, auth_headers):
    content = b"Identical content for dedupe test."
    first = client.post(
        "/v1/documents", headers=auth_headers, files={"file": ("a.txt", content, "text/plain")}
    )
    second = client.post(
        "/v1/documents", headers=auth_headers, files={"file": ("b.txt", content, "text/plain")}
    )
    assert first.status_code == 202
    assert second.status_code == 200  # nothing new accepted: the existing document is returned
    assert first.json()["id"] == second.json()["id"]


def test_get_content_before_ready_returns_409(client, auth_headers):
    # A failed document is never "ready", so /content should refuse it.
    upload = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
    )
    doc_id = upload.json()["id"]
    resp = client.get(f"/v1/documents/{doc_id}/content", headers=auth_headers)
    assert resp.status_code == 409


def test_list_documents_pagination(client, auth_headers):
    for i in range(3):
        client.post(
            "/v1/documents",
            headers=auth_headers,
            files={"file": (f"doc{i}.txt", f"content number {i}".encode(), "text/plain")},
        )
    resp = client.get("/v1/documents?limit=2&offset=0", headers=auth_headers)
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2

    resp2 = client.get("/v1/documents?limit=2&offset=2", headers=auth_headers)
    assert len(resp2.json()["items"]) == 1


def test_delete_document(client, auth_headers):
    upload = client.post(
        "/v1/documents",
        headers=auth_headers,
        files={"file": ("todelete.txt", b"delete me", "text/plain")},
    )
    doc_id = upload.json()["id"]

    delete_resp = client.delete(f"/v1/documents/{doc_id}", headers=auth_headers)
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/v1/documents/{doc_id}", headers=auth_headers)
    assert get_resp.status_code == 404


def test_get_unknown_document_returns_404(client, auth_headers):
    fake_id = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/v1/documents/{fake_id}", headers=auth_headers)
    assert resp.status_code == 404


def test_tenant_isolation_cannot_see_other_tenants_documents(client):
    # Tenant A uploads a document.
    signup_a = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Tenant A", "email": "a@tenanta.io", "password": "supersecret123"},
    )
    headers_a = {"Authorization": f"Bearer {signup_a.json()['token']['access_token']}"}
    upload = client.post(
        "/v1/documents",
        headers=headers_a,
        files={"file": ("secret.txt", b"tenant A confidential data", "text/plain")},
    )
    doc_id = upload.json()["id"]

    # Tenant B tries to access it directly by ID and via listing.
    signup_b = client.post(
        "/v1/auth/signup",
        json={"tenant_name": "Tenant B", "email": "b@tenantb.io", "password": "supersecret123"},
    )
    headers_b = {"Authorization": f"Bearer {signup_b.json()['token']['access_token']}"}

    direct_access = client.get(f"/v1/documents/{doc_id}", headers=headers_b)
    assert direct_access.status_code == 404  # not 403 — existence isn't confirmed either

    listing = client.get("/v1/documents", headers=headers_b)
    assert listing.json()["total"] == 0
