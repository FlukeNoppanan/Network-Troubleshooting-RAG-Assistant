"""No Gemini calls. Reproducible retrieval scores and calibration artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import app


def diagnose():
    model = app.load_embedding_model()
    docs = app.load_documents()
    chunks = app.chunk_documents(docs, tokenizer=model.tokenizer,
                                 max_tokens=model.max_seq_length)
    index, embeddings = app.build_faiss_index(chunks, model)
    cases = pd.read_csv('test_questions.csv').fillna('')
    rows, ospf = [], []
    for case in cases.itertuples():
        results = app.retrieve_chunks(case.question, model, index, chunks)
        accepted = app.relevant_chunks(results)
        correct = (not accepted if case.expected_answer_type == 'NOT_FOUND' else
                   case.expected_source in [r['source'] for r in accepted])
        rows.append({'id': case.id, 'question': case.question,
                     'expected_type': case.expected_answer_type,
                     'expected_source': case.expected_source,
                     'top_source': results[0]['source'],
                     'top_score': round(results[0]['score'], 4),
                     'expected_rank': next((i for i, r in enumerate(results, 1)
                                           if r['source'] == case.expected_source), None),
                     'result': 'PASS' if correct else 'FAIL'})
        if case.id == 15:
            ospf = results
            print('\nOSPF exact regression Top-5:')
            for rank, r in enumerate(results, 1):
                print(rank, r['source'], 'chunk', r['chunk_number'],
                      f"cosine={r['score']:.4f}", r['text'][:200].replace('\n', ' '))
    frame = pd.DataFrame(rows)
    print('\n' + frame.to_string(index=False))
    grounded = frame[frame.expected_type == 'GROUNDED'].top_score
    unrelated = frame[frame.expected_type == 'NOT_FOUND'].top_score
    query = model.encode([cases.iloc[14].question], normalize_embeddings=True)
    token_counts = [len(model.tokenizer.encode(c['text'], add_special_tokens=True,
                                             truncation=False)) for c in chunks]
    stats = {'documents': len(docs), 'raw_characters': sum(len(p.read_text(encoding='utf-8'))
             for p in app.DATA_DIR.glob('*.txt')), 'cleaned_characters': sum(len(d['text']) for d in docs),
             'chunks': len(chunks), 'vectors': index.ntotal, 'dimensions': embeddings.shape[1],
             'max_chunk_tokens': max(token_counts), 'model_token_limit': model.max_seq_length,
             'query_norm': float(np.linalg.norm(query[0])),
             'document_norm_min': float(np.linalg.norm(embeddings, axis=1).min()),
             'document_norm_max': float(np.linalg.norm(embeddings, axis=1).max()),
             'min_grounded_top_score': float(grounded.min()),
             'max_not_found_top_score': float(unrelated.max()), 'threshold': app.RETRIEVAL_THRESHOLD}
    print('\n' + json.dumps(stats, indent=2))
    Path('diagnostics').mkdir(exist_ok=True)
    frame.to_csv('diagnostics/retrieval_results.csv', index=False)
    Path('diagnostics/retrieval_details.json').write_text(
        json.dumps({'stats': stats, 'ospf_top_5': ospf}, ensure_ascii=False, indent=2), encoding='utf-8')
    if (frame.result == 'FAIL').any():
        raise SystemExit('Retrieval regression detected')


if __name__ == '__main__':
    diagnose()
