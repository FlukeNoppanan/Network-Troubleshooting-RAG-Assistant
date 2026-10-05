"""Integration tests with real multilingual embeddings; no Groq API requests."""
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
import app


def set_groq_response(client, content):
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = app.load_documents()
        cls.model = app.load_embedding_model()
        cls.chunks = app.chunk_documents(cls.docs, tokenizer=cls.model.tokenizer,
                                        max_tokens=cls.model.max_seq_length)
        cls.index, cls.embeddings = app.build_faiss_index(cls.chunks, cls.model)

    def test_documents_and_chunking(self):
        self.assertGreaterEqual(len(self.docs), 18)
        self.assertGreater(sum(len(d['text']) for d in self.docs), 35000)
        self.assertTrue(all(sum(c['source'] == d['source'] for c in self.chunks) > 1 for d in self.docs))
        self.assertTrue(all(0 < len(c['text']) <= app.CHUNK_SIZE for c in self.chunks))
        chunks = app.chunk_documents([{'source': 'x', 'text': 'x' * 2500}])
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(self.model.tokenizer.encode(c['text'], truncation=False))
                            <= self.model.max_seq_length for c in self.chunks))
        self.assertIn('\n\n', app.clean_text('หัวข้อ\n\nเนื้อหา'))
        with self.assertRaises(ValueError):
            app.chunk_documents(self.docs, overlap=1000)

    def test_vectors(self):
        self.assertEqual(self.embeddings.shape, (len(self.chunks), self.model.get_sentence_embedding_dimension()))
        self.assertEqual(self.index.ntotal, len(self.chunks))
        np.testing.assert_allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-5)
        query = self.model.encode(['OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร?'],
                                  normalize_embeddings=True).astype('float32')
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-5)
        results = app.retrieve_chunks('OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร?',
                                      self.model, self.index, self.chunks)
        scores = self.embeddings @ query[0]
        order = np.argsort(-scores)[:app.TOP_K]
        for result, i in zip(results, order):
            self.assertEqual(result['source'], self.chunks[i]['source'])
            self.assertEqual(result['chunk_number'], self.chunks[i]['chunk_number'])
            self.assertAlmostEqual(result['score'], float(scores[i]), places=5)

    def test_retrieval_cases(self):
        cases = pd.read_csv(app.DATA_DIR.parent / 'test_questions.csv').fillna('')
        self.assertGreaterEqual(len(cases), 20)
        self.assertGreaterEqual(sum(cases.expected_answer_type == 'NOT_FOUND'), 3)
        for row in cases.itertuples():
            with self.subTest(question=row.question):
                results = app.retrieve_chunks(row.question, self.model, self.index, self.chunks)
                best = results[0]['score']
                print(f'{row.id}: score={best:.3f}, best={results[0]["source"]}')
                if row.expected_answer_type == 'NOT_FOUND':
                    self.assertLess(best, app.RETRIEVAL_THRESHOLD)
                    client = Mock()
                    self.assertEqual(app.generate_answer(row.question, results, client), (app.NOT_FOUND, []))
                    client.chat.completions.create.assert_not_called()
                else:
                    accepted = app.relevant_chunks(results)
                    self.assertTrue(accepted)
                    self.assertIn(row.expected_source, [r['source'] for r in accepted])

    def test_citation_validation(self):
        source = {'source_id': 'S1', 'score': 0.9}
        self.assertEqual(app.validate_answer({'answer': 'Check MTU [S1]', 'citations': ['S1']}, [source])[1], [source])
        valid = app.validate_answer({'answer': 'Check MTU', 'citations': ['S1']}, [source])
        self.assertEqual(valid, ('Check MTU\n\n[S1]', [source]))
        self.assertEqual(app.validate_answer({'answer': app.NOT_FOUND, 'citations': []}, [source]),
                         (app.NOT_FOUND, []))
        for payload in [{'answer': 'Check MTU', 'citations': []},
                        {'answer': 'Check MTU [S9]', 'citations': ['S9']},
                        {'answer': 'text', 'citations': [{}]}, [],
                        {'answer': 'Check MTU [S2]', 'citations': ['S1']}]:
            with self.assertRaises(app.AnswerFormatError):
                app.validate_answer(payload, [source])

    def test_generation_without_network(self):
        sources = app.retrieve_chunks('OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร?',
                                      self.model, self.index, self.chunks)
        self.assertEqual(sources[0]['source'], '05_ospf.txt')
        self.assertTrue(any('MTU mismatch' in r['text'] and 'master/slave' in r['text']
                            for r in app.relevant_chunks(sources)))
        client = Mock()
        set_groq_response(client,
            '{"answer":"ตรวจ MTU mismatch และ DBD master/slave negotiation", "citations":["S1"]}')
        answer, cited = app.generate_answer('OSPF EXSTART', sources, client)
        self.assertNotEqual(answer, app.NOT_FOUND)
        self.assertEqual(cited[0]['source'], '05_ospf.txt')
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['model'], app.GROQ_MODEL)
        self.assertEqual(request['messages'][0]['role'], 'system')
        self.assertIn('ONLY from the provided retrieved context', request['messages'][0]['content'])
        self.assertEqual(request['response_format'], {'type': 'json_object'})
        set_groq_response(client, 'not-json')
        with self.assertRaises(app.AnswerFormatError):
            app.generate_answer('OSPF EXSTART', sources, client)

    def test_ui_grounded_ospf_with_structured_citations(self):
        at = AppTest.from_file(str(app.DATA_DIR.parent / 'app.py'))
        at.secrets['GROQ_API_KEY'] = 'test-only-groq-placeholder'
        at.run(timeout=180)
        client = Mock()
        set_groq_response(client,
            '{"answer":"ตรวจ MTU mismatch และ master/slave negotiation", "citations":["S2"]}')
        manager = Mock()
        manager.__enter__ = Mock(return_value=client)
        manager.__exit__ = Mock(return_value=False)
        groq_module = SimpleNamespace(Groq=Mock(return_value=manager))
        with patch.dict('sys.modules', {'groq': groq_module}):
            at.chat_input[0].set_value('OSPF Neighbor ค้างอยู่ที่ EXSTART เกิดจากอะไร?').run(timeout=30)
        groq_module.Groq.assert_called_once_with(api_key='test-only-groq-placeholder')
        self.assertFalse(at.exception)
        response = at.session_state['messages'][-1]
        self.assertEqual(response['status'], 'grounded_answer')
        self.assertFalse(any(b.key and b.key.startswith('example_') for b in at.button))
        self.assertIn('[S2]', response['content'])
        self.assertEqual(response['sources'][0]['source'], '05_ospf.txt')
        self.assertIn('MTU mismatch', response['sources'][0]['text'])
        self.assertTrue(any(e.label == 'เอกสารอ้างอิง / Retrieved Sources' for e in at.expander))
        at.checkbox[0].check().run()
        self.assertTrue(any(e.label == 'Retrieval Debug' for e in at.expander))
        client.chat.completions.create.assert_called_once()

    def test_ui_welcome_examples(self):
        at = AppTest.from_file(str(app.DATA_DIR.parent / 'app.py'))
        at.secrets['GROQ_API_KEY'] = 'your_groq_api_key_here'
        at.run(timeout=180)
        self.assertFalse(at.exception)
        self.assertEqual(len([b for b in at.button if b.key and b.key.startswith('example_')]), 4)
        self.assertTrue(any(e.label == 'Advanced RAG Details' for e in at.expander))
        at.button(key='example_0').click().run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state['messages'][0]['content'], app.EXAMPLE_QUESTIONS[0])
        self.assertEqual(at.session_state['messages'][-1]['status'], 'missing_secrets')
        self.assertEqual(len(at.session_state['messages']), 2)
        at.run()
        self.assertEqual(len(at.session_state['messages']), 2)
        at.button(key='clear_chat').click().run()
        self.assertEqual(at.session_state['messages'], [])
        self.assertTrue(at.button(key='example_0'))

    def test_ui_missing_secrets_and_chat(self):
        at = AppTest.from_file(str(app.DATA_DIR.parent / 'app.py'))
        # Isolate missing-key UI test from the developer's real configured secret.
        at.secrets['GROQ_API_KEY'] = 'your_groq_api_key_here'
        at.run(timeout=180)
        self.assertFalse(at.exception)
        self.assertTrue(at.warning)
        self.assertEqual(len(at.chat_input), 1)
        at.chat_input[0].set_value('How do I deploy Kubernetes with Helm?').run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state['messages'][-1]['content'], app.NOT_FOUND)
        self.assertEqual(at.session_state['messages'][-1]['sources'], [])
        at.run()
        self.assertEqual(len(at.session_state['messages']), 2)
        at.button(key='clear_chat').click().run()
        self.assertEqual(at.session_state['messages'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
