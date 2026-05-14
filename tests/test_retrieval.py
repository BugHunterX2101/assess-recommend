"""Tests for the retrieval layer."""

import json
import os
import tempfile

import numpy as np
import pytest

from retrieval import embedder


class TestEmbedder:
    def test_encode_returns_numpy_array(self):
        result = embedder.encode(["Test Java developer assessment."])
        assert isinstance(result, np.ndarray)

    def test_encode_returns_float32(self):
        result = embedder.encode(["some text"])
        assert result.dtype == np.float32

    def test_encode_shape(self):
        texts = ["Text one.", "Text two."]
        result = embedder.encode(texts)
        assert result.shape[0] == len(texts)
        assert result.shape[1] > 0

    def test_single_text_encode(self):
        result = embedder.encode(["Single sentence."])
        assert result.shape == (1, result.shape[1])

    def test_encode_semantic_similarity(self):
        """Java-related query should be closer to Java text than Python text."""
        query = embedder.encode(["Java programming developer"])
        java_text = embedder.encode(["Java 8 programming concepts including streams and lambdas"])
        python_text = embedder.encode(["Python data science pandas numpy matplotlib"])

        def cosine_sim(a, b):
            return float(np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))

        sim_java = cosine_sim(query, java_text)
        sim_python = cosine_sim(query, python_text)
        assert sim_java > sim_python


class TestVectorStore:
    def test_load_and_search(self):
        """Build a small FAISS index in memory and test search."""
        from ingestion.indexer import build_index
        from retrieval import vector_store

        catalog = [
            {
                "id": "java-8-new",
                "name": "Java 8 (New)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                "test_type": ["Knowledge & Skills"],
                "description": "Measures Java 8 programming concepts.",
            },
            {
                "id": "python-new",
                "name": "Python (New)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
                "test_type": ["Knowledge & Skills"],
                "description": "Assesses Python programming proficiency.",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = os.path.join(tmpdir, "faiss.index")
            meta_path = os.path.join(tmpdir, "catalog_metadata.json")

            build_index(catalog, index_path=idx_path, metadata_path=meta_path)
            vector_store.load(index_path=idx_path, metadata_path=meta_path)

            query_vec = embedder.encode(["Java developer assessment"])
            results = vector_store.search(query_vec[0], k=2)

            assert len(results) > 0
            assert results[0]["name"] in {"Java 8 (New)", "Python (New)"}

    def test_get_all_urls(self):
        from retrieval import vector_store
        urls = vector_store.get_all_urls()
        assert isinstance(urls, set)
