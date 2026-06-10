from __future__ import annotations

import json

from scripts.coral_detection_preview_server import load_labels


def test_load_pbtxt_labels_prefers_display_name(tmp_path):
    label_map = tmp_path / "labels.pbtxt"
    label_map.write_text(
        """
        item {
          name: "/m/01g317"
          id: 1
          display_name: "person"
        }
        item { name: "/m/0k4j" id: 3 display_name: "car" }
        """,
        encoding="utf-8",
    )

    assert load_labels(label_map) == {1: "person", 3: "car"}


def test_load_json_labels_remains_supported(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"1": "person", "3": "car"}), encoding="utf-8")

    assert load_labels(labels) == {1: "person", 3: "car"}
