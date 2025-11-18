import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call
import sys
from pathlib import Path
import platform
import subprocess
import requests

# Add the parent directory to the Python path to allow module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
except ImportError:  # Provide a lightweight stub so tests can run without PyTorch
    torch = MagicMock()
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    sys.modules['torch'] = torch

try:
    import langchain_mcp_adapters.client  # noqa: F401
except ImportError:
    mock_mcp_pkg = MagicMock()
    mock_mcp_client_module = MagicMock()
    mock_mcp_client_module.MultiServerMCPClient = MagicMock()
    mock_mcp_client_module.StdioConnection = MagicMock()
    mock_mcp_pkg.client = mock_mcp_client_module
    sys.modules['langchain_mcp_adapters'] = mock_mcp_pkg
    sys.modules['langchain_mcp_adapters.client'] = mock_mcp_client_module

try:
    import networkx as nx  # noqa: F401
except ImportError:
    nx = MagicMock()
    sys.modules['networkx'] = nx

try:
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
except ImportError:
    sklearn_stub = MagicMock()
    metrics_stub = MagicMock()
    pairwise_stub = MagicMock()
    pairwise_stub.cosine_similarity = MagicMock()
    metrics_stub.pairwise = pairwise_stub
    sklearn_stub.metrics = metrics_stub
    sys.modules['sklearn'] = sklearn_stub
    sys.modules['sklearn.metrics'] = metrics_stub
    sys.modules['sklearn.metrics.pairwise'] = pairwise_stub

try:
    import nltk  # noqa: F401
except ImportError:
    nltk_stub = MagicMock()
    nltk_stub.data = MagicMock()
    nltk_stub.data.load = MagicMock(side_effect=LookupError('punkt not found'))
    nltk_stub.download = MagicMock()
    sys.modules['nltk'] = nltk_stub

from Tepora_app.agent_core.memory.memory_system import MemorySystem
from Tepora_app.agent_core.llm_manager import LLMManager
from Tepora_app.agent_core.tool_manager import ToolManager
from Tepora_app.agent_core.em_llm_core import EMEventSegmenter, EMConfig, EMLLMIntegrator
from Tepora_app.agent_core.graph import AgentCore
from Tepora_app.agent_core import config as agent_config

class TestLLMManager(unittest.TestCase):

    def setUp(self):
        """Set up a fresh LLMManager instance for each test."""
        # Mock the config module to have predictable values
        self.mock_config = MagicMock()
        self.mock_config.MODELS_GGUF = {
            "gemma_3n": {
                "port": 8000,
                "path": "models/gemma-3n.gguf",
                "n_ctx": 4096,
                "n_gpu_layers": -1,
                "logprobs": True,
            },
            "embedding_model": {
                "port": 8003,
                "path": "models/embedding.gguf",
                "n_ctx": 4096,
                "n_gpu_layers": -1,
            },
        }
        self.mock_config.LLAMA_CPP_CONFIG = {
            "health_check_timeout": 2,
            "health_check_interval": 0.1,
            "process_terminate_timeout": 1,
            "embedding_health_check_timeout": 2,
        }

        # Patch the config module used by llm_manager
        self.config_patcher = patch('Tepora_app.agent_core.llm_manager.config', self.mock_config)
        self.mock_config_instance = self.config_patcher.start()

        self.manager = LLMManager()

    def tearDown(self):
        """Clean up by stopping the patcher."""
        self.config_patcher.stop()
        # Ensure cleanup is called, but only if a process was mocked
        if self.manager._active_process or self.manager._embedding_process:
            self.manager.cleanup()

    def test_initialization(self):
        """Test that the LLMManager initializes with default empty values."""
        self.assertIsNone(self.manager._current_model_key)
        self.assertIsNone(self.manager._chat_llm)
        self.assertIsNone(self.manager._active_process)
        self.assertIsNone(self.manager._embedding_llm)
        self.assertIsNone(self.manager._embedding_process)

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.stat')
    def test_find_server_executable_prefers_version_and_env(self, mock_stat, mock_glob):
        """Test the server executable finding logic."""
        # Mock file stats (mtime)
        mock_stat.return_value.st_mtime = 1.0

        # Mock found files
        mock_files = [
            Path("llama.cpp/llama-b100-bin-win-cpu-x64/llama-server.exe"),
            Path("llama.cpp/llama-b200-bin-win-cuda-x64/llama-server.exe"), # Highest version and best env
            Path("llama.cpp/llama-b150-bin-win-vulkan-x64/llama-server.exe"),
        ]
        mock_glob.return_value = mock_files

        with patch('torch.cuda.is_available', return_value=True), \
             patch('sys.platform', "win32"):
            result = self.manager._find_server_executable(Path("llama.cpp/"))
            self.assertEqual(result, mock_files[1])

    @patch('requests.get')
    def test_perform_health_check_success(self, mock_get):
        """Test health check success on the first try."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        try:
            self.manager._perform_health_check(8000, "test_model")
        except TimeoutError:
            self.fail("_perform_health_check raised TimeoutError unexpectedly.")
        mock_get.assert_called_once_with("http://localhost:8000/health", timeout=0.5)

    @patch('requests.get')
    def test_perform_health_check_retry_and_fail(self, mock_get):
        """Test that health check retries and eventually fails."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

        # We need to reduce the timeout for the test to run quickly
        self.mock_config_instance.LLAMA_CPP_CONFIG['health_check_timeout'] = 3  # max_retries = 3
        self.mock_config_instance.LLAMA_CPP_CONFIG['health_check_interval'] = 0.1

        with self.assertRaises(TimeoutError):
            self.manager._perform_health_check(8000, "test_model")

        # Check that it was called multiple times
        self.assertGreater(mock_get.call_count, 1)

    @patch('Tepora_app.agent_core.llm_manager.launch_server')
    @patch('pathlib.Path.exists', return_value=True)
    @patch.object(LLMManager, '_find_server_executable', return_value=Path("path/to/server"))
    @patch.object(LLMManager, '_perform_health_check', return_value=None)
    @patch('Tepora_app.agent_core.llm_manager.ChatOpenAI')
    def test_load_model_success(self, mock_chat_openai, mock_health_check, mock_find_server, mock_path_exists, mock_launch_server):
        """Test the successful loading of a chat model."""
        mock_process = MagicMock()
        mock_launch_server.return_value = mock_process

        # Action
        llm = self.manager.get_character_agent()

        # Assertions
        mock_find_server.assert_called_once()
        mock_launch_server.assert_called_once()
        mock_health_check.assert_called_once_with(8000, "gemma_3n", unittest.mock.ANY)
        mock_chat_openai.assert_called_once()
        self.assertIsNotNone(llm)
        self.assertEqual(self.manager._current_model_key, "gemma_3n")
        self.assertEqual(self.manager._active_process, mock_process)

    @patch('Tepora_app.agent_core.llm_manager.launch_server')
    @patch('pathlib.Path.exists', return_value=True)
    @patch.object(LLMManager, '_find_server_executable', return_value=Path("path/to/server"))
    @patch.object(LLMManager, '_perform_health_check', return_value=None)
    @patch('Tepora_app.agent_core.llm_manager.OpenAIEmbeddings')
    def test_get_embedding_model_success(self, mock_embeddings, mock_health_check, mock_find_server, mock_path_exists, mock_launch_server):
        """Test the successful loading of the embedding model."""
        mock_process = MagicMock()
        mock_launch_server.return_value = mock_process

        # Action
        embedding_llm = self.manager.get_embedding_model()

        # Assertions
        mock_launch_server.assert_called_once()
        command = mock_launch_server.call_args[0][0]
        self.assertIn("--embedding", command)
        mock_health_check.assert_called_once_with(8003, "embedding_model", unittest.mock.ANY)
        mock_embeddings.assert_called_once()
        self.assertIsNotNone(embedding_llm)
        self.assertIsNotNone(self.manager._embedding_llm)
        self.assertEqual(self.manager._embedding_process, mock_process)

        # Test caching
        embedding_llm2 = self.manager.get_embedding_model()
        self.assertIs(embedding_llm, embedding_llm2)
        mock_launch_server.assert_called_once()  # Should not be called again

    def test_cleanup(self):
        """Test that the cleanup method properly unloads all models."""
        # Mock loaded models and processes
        mock_chat_process = MagicMock(spec=subprocess.Popen)
        mock_chat_process.pid = 1234
        self.manager._active_process = mock_chat_process

        mock_embedding_process = MagicMock(spec=subprocess.Popen)
        mock_embedding_process.pid = 5678
        self.manager._embedding_process = mock_embedding_process

        self.manager._current_model_key = "gemma_3n"
        self.manager._embedding_llm = MagicMock()

        # Action
        self.manager.cleanup()

        # Assertions
        mock_chat_process.terminate.assert_called_once()
        mock_embedding_process.terminate.assert_called_once()
        self.assertIsNone(self.manager._active_process)
        self.assertIsNone(self.manager._embedding_process)
        self.assertIsNone(self.manager._current_model_key)
        self.assertIsNone(self.manager._chat_llm)
        self.assertIsNone(self.manager._embedding_llm)

    @patch.object(LLMManager, '_load_model')
    def test_model_switching(self, mock_load_model):
        """Test that calling different agent getters switches the model."""
        # Mock config for the second model
        self.mock_config.MODELS_GGUF['jan_nano'] = {"port": 8001, "path": "models/jan.gguf"}

        # First call
        self.manager.get_character_agent()
        mock_load_model.assert_called_with("gemma_3n")
        # Simulate state change after loading
        self.manager._current_model_key = "gemma_3n"

        # Second call
        self.manager.get_professional_agent()
        mock_load_model.assert_called_with("jan_nano")
        self.manager._current_model_key = "jan_nano"

        self.assertEqual(mock_load_model.call_count, 2)

    def test_token_counting_with_empty_messages(self):
        """Test that token counting returns 0 for an empty list of messages."""
        count = self.manager.count_tokens_for_messages([])
        self.assertEqual(count, 0)


class TestMemorySystem(unittest.TestCase):

    def setUp(self):
        embedding_provider = MagicMock()
        embedding_provider.encode.return_value = [[0.1, 0.2, 0.3]]
        self.embedding_provider = embedding_provider

        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            self.memory_system = MemorySystem(self.embedding_provider, db_path="./test_db", collection_name="test_memory")
            self.collection = mock_collection

    def test_retrieve_handles_missing_distances(self):
        query_embedding = [[0.5, 0.6, 0.7]]
        self.embedding_provider.encode.return_value = query_embedding

        self.collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'distances': None,
            'metadatas': [[{'created_ts': 100, 'history_json': '{}', 'metadata_json': '{}'},
                           {'created_ts': 50, 'history_json': '{}', 'metadata_json': '{}'}]],
            'documents': [['summary1', 'summary2']]
        }

        results = self.memory_system.retrieve("test-query", k=2)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('score', result)

    def test_retrieve_handles_short_distance_list(self):
        query_embedding = [[0.5, 0.6, 0.7]]
        self.embedding_provider.encode.return_value = query_embedding

        self.collection.query.return_value = {
            'ids': [['id1', 'id2', 'id3']],
            'distances': [[0.0, 0.1, 0.2]],
            'metadatas': [[{'created_ts': 100, 'history_json': '{}', 'metadata_json': '{}'},
                           {'created_ts': 90, 'history_json': '{}', 'metadata_json': '{}'},
                           {'created_ts': 80, 'history_json': '{}', 'metadata_json': '{}'}]],
            'documents': [['summary1', 'summary2', 'summary3']]
        }

        results = self.memory_system.retrieve("test-query", k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['id'], 'id1')
        self.collection.query.assert_called_once()


class TestToolManager(unittest.TestCase):

    def test_cleanup_stops_loop_and_closes_resources(self):
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mock_loop.is_closed.return_value = False
        mock_loop.shutdown_asyncgens.return_value = 'shutdown-coro'

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        future_shutdown = MagicMock()
        future_shutdown.result.return_value = None

        with patch('Tepora_app.agent_core.tool_manager.asyncio.new_event_loop', return_value=mock_loop), \
             patch('Tepora_app.agent_core.tool_manager.threading.Thread', return_value=mock_thread) as mock_thread_cls, \
             patch('Tepora_app.agent_core.tool_manager.asyncio.run_coroutine_threadsafe') as mock_run_coroutine:

            mock_run_coroutine.return_value = future_shutdown

            tool_manager = ToolManager("dummy_config.json")

            tool_manager.cleanup()

            # Async generator shutdown path
            mock_run_coroutine.assert_called_once_with(mock_loop.shutdown_asyncgens.return_value, mock_loop)
            mock_run_coroutine.assert_called_with(mock_loop.shutdown_asyncgens.return_value, mock_loop)
            future_shutdown.result.assert_called_once_with(timeout=5)

            mock_loop.call_soon_threadsafe.assert_called_once_with(mock_loop.stop)
            mock_thread.join.assert_called_once_with(timeout=5)
            mock_loop.close.assert_called_once()

    def test_initialize_uses_helper_loaders(self):
        mock_loop = MagicMock()
        mock_thread = MagicMock()

        with patch('Tepora_app.agent_core.tool_manager.asyncio.new_event_loop', return_value=mock_loop), \
             patch('Tepora_app.agent_core.tool_manager.threading.Thread', return_value=mock_thread), \
             patch('Tepora_app.agent_core.tool_manager.load_native_tools', return_value=['native_tool']) as mock_load_native, \
             patch('Tepora_app.agent_core.tool_manager.load_mcp_tools_robust', return_value=['mcp_tool']) as mock_load_mcp:

            tool_manager = ToolManager("dummy_config.json")
            tool_manager.initialize()

            mock_load_native.assert_called_once()
            mock_load_mcp.assert_called_once_with(tool_manager.config_path)
            self.assertEqual(tool_manager.tools, ['native_tool', 'mcp_tool'])
            self.assertEqual(tool_manager.tool_map, {'native_tool': 'native_tool', 'mcp_tool': 'mcp_tool'})


class TestEMEventSegmenter(unittest.TestCase):

    def test_fallback_tokenizer_used_when_punkt_unavailable(self):
        with patch('Tepora_app.agent_core.em_llm_core.nltk.data.load', side_effect=LookupError('missing data')) as mock_load, \
             patch('Tepora_app.agent_core.em_llm_core.nltk.download', side_effect=Exception('offline')) as mock_download:

            segmenter = EMEventSegmenter(EMConfig())

        mock_load.assert_called_once_with('tokenizers/punkt/english.pickle')
        mock_download.assert_called_once_with('punkt', quiet=True)
        self.assertEqual(segmenter.sent_tokenizer.__class__.__name__, '_SimpleSentenceTokenizer')

        sentences = segmenter._split_into_sentences("Hello world. Next line!\nFinal question?")
        self.assertGreaterEqual(len(sentences), 2)
        self.assertIn('Hello world', sentences[0])


class TestEMLLMIntegrator(unittest.IsolatedAsyncioTestCase):

    async def test_process_logprobs_skips_invalid_entries(self):
        tokenizer_stub = MagicMock()
        tokenizer_stub.tokenize.side_effect = lambda text: text.split()

        with patch('Tepora_app.agent_core.em_llm_core.EMEventSegmenter._get_sentence_tokenizer', return_value=tokenizer_stub):
            memory_system = MagicMock()
            integrator = EMLLMIntegrator(
                llm_manager=MagicMock(),
                embedding_provider=MagicMock(),
                config=EMConfig(),
                memory_system=memory_system,
            )

        integrator.segmenter.calculate_surprise_from_logprobs = MagicMock(return_value=[0.3, 0.2])
        integrator.segmenter._identify_event_boundaries = MagicMock(return_value=[0, 2])

        captured = {}

        async def fake_finalize(events):
            captured['events'] = events
            return events

        integrator._finalize_and_store_events = AsyncMock(side_effect=fake_finalize)

        logprobs_payload = [
            {'token': 'token_a', 'logprob': -0.1},
            {'token': 'token_missing_logprob'},
            {'logprob': -0.2},
            {'token_str': 'token_b', 'logprob': -0.3},
        ]

        results = await integrator.process_logprobs_for_memory(logprobs_payload)

        integrator.segmenter.calculate_surprise_from_logprobs.assert_called_once()
        passed_entries = integrator.segmenter.calculate_surprise_from_logprobs.call_args[0][0]
        self.assertEqual(len(passed_entries), 2)
        self.assertTrue(all('token' in entry for entry in passed_entries))
        self.assertEqual(passed_entries[1]['token'], 'token_b')

        integrator.segmenter._identify_event_boundaries.assert_called_once_with([0.3, 0.2], ['token_a', 'token_b'])
        integrator._finalize_and_store_events.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(captured['events'][0].tokens, ['token_a', 'token_b'])


class TestAgentCoreRouting(unittest.TestCase):

    def setUp(self):
        self.core = AgentCore(
            llm_manager=MagicMock(),
            tool_manager=MagicMock(),
            memory_system=MagicMock(),
        )

    def test_route_stats_commands(self):
        state = {"input": "/emstats"}
        self.assertEqual(self.core.route_by_command(state), "stats")

        state = {"input": "   /emstats_prof"}
        self.assertEqual(self.core.route_by_command(state), "stats")

        state = {"input": " /emstats_char latest"}
        self.assertEqual(self.core.route_by_command(state), "stats")

    def test_route_defaults_to_direct_answer(self):
        state = {"input": "hello there"}
        self.assertEqual(self.core.route_by_command(state), "direct_answer")


if __name__ == '__main__':
    unittest.main()
