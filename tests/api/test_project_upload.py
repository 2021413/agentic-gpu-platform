"""A project is created from the files it is made of, at a path nobody chose.

Every project used to be registered with `local_path` and every one of them
said `/projects/current`, so the sidebar offered fifteen names for one mount.
Selecting a project ran the agents on whatever was there. This route is the
replacement, and the old route no longer takes a path at all.
"""

from __future__ import annotations

import io
import zipfile

import httpx
from tests.api.conftest import Harness


def zipped(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


async def upload(client: httpx.AsyncClient, name: str, **fields: str) -> httpx.Response:
    return await client.post(
        "/v1/projects/upload",
        data={"name": name, **fields},
        files=[
            ("files", ("pyproject.toml", b"[project]\nname = 'demo'\n")),
            ("files", ("src/demo.py", b"def f():\n    return 1\n")),
        ],
    )


# -- the happy paths ---------------------------------------------------------
async def test_several_files_become_a_project(client: httpx.AsyncClient) -> None:
    response = await upload(client, "demo")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "demo"
    assert "local_path" not in body, "where the files live is not the client's business"


async def test_the_toolchain_is_detected_from_the_files(client: httpx.AsyncClient) -> None:
    """The caller sent a pyproject and no commands; the server should know
    what to do with a Python project without being told."""
    body = (await upload(client, "demo")).json()

    assert body["toolchain"]["language"] == "python"
    assert body["toolchain"]["test_command"] == "pytest -q"


async def test_a_given_command_wins_over_detection_field_by_field(
    client: httpx.AsyncClient,
) -> None:
    """A caller who knows the test command but not the language must not have
    to guess the language to keep the command."""
    body = (await upload(client, "demo", test_command="pytest tests/unit -q")).json()

    assert body["toolchain"]["test_command"] == "pytest tests/unit -q"
    assert body["toolchain"]["language"] == "python", "still detected"
    assert body["toolchain"]["build_command"] == "python -m compileall -q .", "still detected"


async def test_a_zip_is_extracted(client: httpx.AsyncClient) -> None:
    blob = zipped({"app/package.json": b"{}", "app/index.js": b"", "app/README": b""})

    response = await client.post(
        "/v1/projects/upload",
        data={"name": "js"},
        files=[("files", ("app.zip", blob))],
    )

    assert response.status_code == 201, response.text
    assert response.json()["toolchain"]["language"] == "javascript", (
        "the single top-level folder was stripped, so package.json is at the root"
    )


async def test_the_project_is_then_listed_and_selectable(client: httpx.AsyncClient) -> None:
    created = (await upload(client, "demo")).json()

    listed = (await client.get("/v1/projects")).json()

    assert [p["id"] for p in listed] == [created["id"]]


# -- the refusals ------------------------------------------------------------
async def test_a_second_upload_under_the_same_name_is_a_conflict(
    client: httpx.AsyncClient,
) -> None:
    """Creation by reference is idempotent on the name because the same
    reference means the same code. An upload is not: answering with the old
    project would run the agents on code the caller did not send."""
    await upload(client, "demo")

    response = await upload(client, "demo")

    assert response.status_code == 409
    assert response.json()["code"] == "project_exists"


async def test_a_corrupt_archive_is_a_400_with_the_reason(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/projects/upload",
        data={"name": "broken"},
        files=[("files", ("x.zip", b"this is not a zip"))],
    )

    assert response.status_code == 400
    assert "zip" in response.json()["detail"]


async def test_a_path_that_escapes_is_refused_and_nothing_is_created(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    response = await client.post(
        "/v1/projects/upload",
        data={"name": "evil"},
        files=[("files", ("../../etc/cron.d/x", b"* * * * * root true\n"))],
    )

    assert response.status_code == 400
    assert (await client.get("/v1/projects")).json() == [], "a refused upload leaves no record"


async def test_the_name_is_required(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/projects/upload", files=[("files", ("a.py", b"pass\n"))]
    )

    assert response.status_code == 422


# -- the old door is shut ----------------------------------------------------
async def test_the_json_route_no_longer_accepts_a_path(client: httpx.AsyncClient) -> None:
    """`local_path` was the whole problem. It is not merely ignored, it is a
    422: a client that still sends it must learn it is gone, not have it
    silently dropped and a project created against nothing."""
    response = await client.post(
        "/v1/projects",
        json={"name": "old", "local_path": "/projects/current", "default_branch": "main"},
    )

    assert response.status_code == 422
