from __future__ import annotations

import argparse
import json
import zipfile

from scripts.organize_rna_data import ARCHIVE_LAYOUT, EXTERNAL_MOUSE_LAYOUT, organize


def test_organizer_creates_canonical_english_layout(tmp_path):
    delivery = tmp_path / "source_delivery.zip"
    with zipfile.ZipFile(delivery, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, basename in enumerate(ARCHIVE_LAYOUT):
            archive.writestr(f"delivered_data/{basename}", f"archive-{index}".encode())
        archive.writestr("delivered_data/handoff_README.md", b"delivery notes")

    external = {}
    for index, key in enumerate(EXTERNAL_MOUSE_LAYOUT):
        path = tmp_path / f"input_{key}.csv.gz"
        path.write_bytes(f"mouse-{index}".encode())
        external[key] = path

    output = tmp_path / "rna_mamba_data"
    args = argparse.Namespace(
        delivery_archive=str(delivery),
        mouse_transcript_master=str(external["mouse_transcript_master"]),
        mouse_m6a_mask=str(external["mouse_m6a_mask"]),
        mouse_exon_map=str(external["mouse_exon_map"]),
        output_dir=str(output),
        overwrite=False,
    )
    manifest = organize(args)

    assert manifest["pretraining"]["non_coding_target"] == 3_000_000
    assert manifest["pretraining"]["coding_raw_records"] == 1_953_101
    for relative_path in (*ARCHIVE_LAYOUT.values(), *EXTERNAL_MOUSE_LAYOUT.values()):
        assert (output / relative_path).is_file()
    assert (output / "documentation" / "delivery_readme.md").is_file()

    inventory = json.loads(
        (output / "manifests" / "source_inventory.json").read_text(encoding="utf-8")
    )
    assert all(not item["path"].startswith(str(tmp_path)) for item in inventory["files"])
    assert all(".textClipping" not in item["path"] for item in inventory["files"])
