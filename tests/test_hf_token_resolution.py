"""
Unit Tests for Hugging Face Hub Authentication & Token Resolution Engine
=========================================================================
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.hf_utils import _clean_token, get_hf_token, setup_hf_token


class TestHFTokenResolution(unittest.TestCase):
    def setUp(self):
        # Save original environment
        self._orig_env = {
            k: os.environ.get(k)
            for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }
        for k in self._orig_env:
            os.environ.pop(k, None)

    def tearDown(self):
        # Restore original environment
        for k, v in self._orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_clean_token(self):
        self.assertIsNone(_clean_token(None))
        self.assertIsNone(_clean_token(""))
        self.assertIsNone(_clean_token("   "))
        self.assertIsNone(_clean_token("none"))
        self.assertIsNone(_clean_token("null"))
        self.assertEqual(_clean_token("hf_valid_token_123"), "hf_valid_token_123")
        self.assertEqual(_clean_token("  'hf_quoted_token'  "), "hf_quoted_token")
        self.assertEqual(_clean_token('  "hf_double_quoted"  '), "hf_double_quoted")

    def test_explicit_token_highest_priority(self):
        os.environ["HF_TOKEN"] = "env_token_ignore"
        tok = get_hf_token(explicit_token="explicit_token_win")
        self.assertEqual(tok, "explicit_token_win")

    def test_env_var_pickup(self):
        os.environ["HF_TOKEN"] = "hf_env_secret_token"
        tok = get_hf_token()
        self.assertEqual(tok, "hf_env_secret_token")

    def test_website_vault_pickup(self):
        with patch("src.hf_utils.fetch_hf_token_from_website", return_value="hf_vault_secret"):
            tok = get_hf_token()
            self.assertEqual(tok, "hf_vault_secret")

    def test_colab_userdata_pickup(self):
        mock_userdata = MagicMock()
        mock_userdata.get.return_value = "hf_colab_secret"
        with patch("src.hf_utils.fetch_hf_token_from_website", return_value=None):
            with patch.dict("sys.modules", {"google.colab": MagicMock(userdata=mock_userdata)}):
                tok = get_hf_token()
                self.assertEqual(tok, "hf_colab_secret")

    def test_kaggle_secrets_pickup(self):
        mock_client = MagicMock()
        mock_client.return_value.get_secret.return_value = "hf_kaggle_secret"
        with patch("src.hf_utils.fetch_hf_token_from_website", return_value=None):
            with patch.dict("sys.modules", {"kaggle_secrets": MagicMock(UserSecretsClient=mock_client)}):
                tok = get_hf_token()
                self.assertEqual(tok, "hf_kaggle_secret")

    def test_setup_hf_token_exports_and_masks(self):
        with patch("huggingface_hub.login") as mock_login:
            res = setup_hf_token(explicit_token="hf_test_token_abcdef1234", verbose=True)
            self.assertEqual(res, "hf_test_token_abcdef1234")
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_test_token_abcdef1234")
            self.assertEqual(os.environ.get("HUGGING_FACE_HUB_TOKEN"), "hf_test_token_abcdef1234")
            mock_login.assert_called_once_with(token="hf_test_token_abcdef1234", add_to_git_credential=False)

    def test_corpus_and_elastic_window_delegation(self):
        from src.corpus import _get_hf_token as corpus_tok
        from src.elastic_window import _get_hf_token as elastic_tok

        os.environ["HF_TOKEN"] = "hf_unified_token"
        self.assertEqual(corpus_tok(), "hf_unified_token")
        self.assertEqual(elastic_tok(), "hf_unified_token")


if __name__ == "__main__":
    unittest.main()
