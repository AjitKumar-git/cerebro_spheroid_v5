from src import elastic_window


def test_raw_document_keeps_first_paragraph_in_loss_target():
    document = "Opening paragraph.\n\nSecond paragraph with #### as source text, unchanged."

    prompt, target = elastic_window.extract_prompt_response_pair({"text": document})

    assert prompt == "Document:\n"
    assert target == document


def test_fineweb_alias_selects_small_sample_and_pinned_revision(monkeypatch):
    calls = []

    def fake_open(repo, config_name=None, **kwargs):
        calls.append((repo, config_name, kwargs))
        return iter([{"text": "A complete FineWeb document.", "id": "doc-1"}])

    monkeypatch.setattr(elastic_window, "_open_hf_stream", fake_open)
    stream = elastic_window.build_online_dataset_iterator(
        dataset_source="fineweb", inject_retrieval=False
    )

    row = next(stream)

    assert row["_dataset"] == "HuggingFaceFW/fineweb"
    assert calls == [
        (
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            {
                "rank": 0,
                "world_size": 1,
                "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
            },
        )
    ]

