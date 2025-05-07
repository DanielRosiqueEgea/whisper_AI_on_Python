import unittest
from unittest.mock import Mock, patch, mock_open
import os
import json

from utils import transcribe_chunks  # Asegúrate de usar el nombre real del módulo

class TestTranscribeChunks(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("json.dump")
    @patch("json.load")
    def test_transcribe_chunks_reads_existing_json(
        self, mock_json_load, mock_json_dump, mock_exists, mock_open_file
    ):
        model = Mock()
        adjust_segments = Mock()

        # Setup
        audio_stem = "audio"
        output_dir = "/fake/output"
        all_segments = []

        # Fake segments
        existing_segments = [{"start": 0, "end": 1, "text": "Hola"}]
        mock_json_load.return_value = existing_segments

        chunk_path = "/fake/audio_chunk_0.mp3"
        chunks = [(chunk_path, 0)]

        mock_exists.return_value = True  # Simula que el archivo .json ya existe

        transcribe_chunks(model, adjust_segments, audio_stem, output_dir, all_segments, chunks)

        self.assertEqual(all_segments, existing_segments)
        mock_json_load.assert_called_once()
        model.transcribe.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("json.dump")
    def test_transcribe_chunks_transcribes_if_json_missing(
        self, mock_json_dump, mock_exists, mock_open_file
    ):
        model = Mock()
        adjust_segments = Mock()

        audio_stem = "audio"
        output_dir = "/fake/output"
        all_segments = []

        chunks = [("/fake/audio_chunk_60.mp3", 60)]
        mock_exists.return_value = False

        # Simular salida de Whisper
        fake_result = {"segments": [{"start": 60, "end": 61, "text": "Test"}]}
        model.transcribe.return_value = fake_result

        adjusted_segments = [{"start": 61, "end": 62, "text": "Adjusted"}]
        adjust_segments.return_value = adjusted_segments

        transcribe_chunks(model, adjust_segments, audio_stem, output_dir, all_segments, chunks)

        self.assertEqual(all_segments, adjusted_segments)
        model.transcribe.assert_called_once_with("/fake/audio_chunk_60.mp3", language="Spanish", verbose=None)
        mock_json_dump.assert_called_once_with(adjusted_segments, mock_open_file(), ensure_ascii=False, indent=2)

# TODO: Añadir todos los demás tests

if __name__ == "__main__":
    unittest.main(verbosity=2)
