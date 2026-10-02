"""Screen new original controls against visible reserved inputs; never select gold."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from jev.data import read_split_directory
from scripts.screen_training_overlap import build_index, fingerprint, leaves, norm, shingles


JOURNAL_FIELDS = {
    'classify_banking77': ('text',), 'prompt_injection': ('text',),
    'policy': (), 'guardrail': ('tool',), 'nl_filter': (),
    'chess_openjev_vs_random': ('fen', 'legal'),
    'poker_openjev_vs_random': ('hole', 'board', 'options'),
    'poker_openjev_vs_rulebot': ('hole', 'board', 'options'),
    'poker_openjev_vs_rulebot_60': ('hole', 'board', 'options'), 'snake': ('options',),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(dataset, benchmark, routing_root, raw_dir, source_manifests):
    dataset, benchmark, routing_root, raw_dir = map(Path, (dataset, benchmark, routing_root, raw_dir))
    sources = {}
    for manifest in source_manifests:
        document = json.loads(Path(manifest).read_text())
        entries = document if isinstance(document, list) else document['sources']
        for source in entries:
            name = source['snapshot']
            if Path(name).name != name:
                raise ValueError('Snapshot paths must be local filenames')
            path = raw_dir/name
            if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
                raise ValueError('Source snapshot changed: '+name)
            sources[name] = source['sha256']
    document = json.loads(benchmark.read_text())
    if len(document['workloads']) != 231:
        raise ValueError('Expected all 231 public development input projections')
    auxiliary, inventory = [], {'jevbench_public_inputs': 231}

    def add_input(value):
        if list(leaves(value)):
            auxiliary.append({'request': {'state': value, 'questions': {'visible_only': {
                'type': 'noul', 'instructions': 'Visible-input lexical screening only.'}}}})

    for split in ('test', 'calibration'):
        path = routing_root/(split+'-inputs.csv')
        with path.open(newline='') as handle:
            inputs = list(csv.DictReader(handle))
        for item in inputs:
            add_input(item['text'])
        inventory['natural_routing_'+split] = len(inputs)
    for name, fields in JOURNAL_FIELDS.items():
        path = raw_dir/('author-demos--out--'+name+'--decisions.jsonl')
        if path.name not in sources:
            raise ValueError('Missing journal source binding: '+path.name)
        journal = [json.loads(line) for line in path.read_text().splitlines()]
        for item in journal:
            add_input({field: item[field] for field in fields})
        inventory['upstream_journal_'+name] = {'rows': len(journal), 'selected_visible_fields': list(fields)}
    literal_count = 0
    for name in ('policy', 'guardrail', 'nl_filter', 'classify', 'prompt_injection'):
        path = raw_dir/('author-demos--'+name+'.py')
        if path.name not in sources:
            raise ValueError('Missing source-literal binding: '+path.name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                add_input(node.value)
                literal_count += 1
    inventory['upstream_demo_source_string_literals'] = literal_count
    for name in ('browser_hf', 'browser_hn2', 'browser_wikipedia'):
        path = raw_dir/('author-demos--out--'+name+'--run.json')
        if path.name not in sources:
            raise ValueError('Missing browser summary binding: '+path.name)
        run = json.loads(path.read_text())
        add_input({key: run[key] for key in ('task', 'start_url')})
    inventory['upstream_browser_task_descriptions'] = 3
    exact, grams, short, requests = build_index([document, {'workloads': auxiliary}])
    rows = list(read_split_directory(dataset))
    manifest = json.loads((dataset/'manifest.json').read_text())
    if any(sha(dataset/name) != expected for name, expected in manifest['files_sha256'].items()):
        raise ValueError('Candidate split changed during visible-input screening')
    matches = []
    for row in rows:
        reasons = []
        if fingerprint(row) in exact:
            reasons.append('exact_visible_input')
        text = list(leaves(row['state']))
        if any(norm(value) in short for value in text):
            reasons.append('normalized_state_leaf_ge8_words')
        if any(shingles(value) & grams for value in text):
            reasons.append('state_13_token_shingle')
        if reasons:
            matches.append({'id': row['id'], 'group_id': row['group_id'],
                            'split': row['split'], 'reasons': reasons})
    bindings = [{'name': path.name, 'sha256': sha(path)} for path in
                [benchmark, routing_root/'test-inputs.csv', routing_root/'calibration-inputs.csv']]
    return {'status': 'no_detected_overlap' if not matches else 'overlap_detected',
        'dataset_manifest_sha256': sha(dataset/'manifest.json'), 'split_sha256': manifest['files_sha256'],
        'screen_script_sha256': sha(__file__), 'lexical_screen_sha256': sha(Path(__file__).with_name('screen_training_overlap.py')),
        'compile_api_sha256': sha(Path(__file__).resolve().parents[1]/'jev/api.py'),
        'candidate_rows_checked': len(rows), 'source_snapshot_sha256': sources,
        'source_manifest_sha256': {Path(path).name: sha(path) for path in source_manifests},
        'input_bindings': bindings, 'visible_corpus_inventory': inventory, 'compiled_visible_requests': requests,
        'matched_rows': len(matches), 'quarantined_groups': sorted({row['group_id'] for row in matches}),
        'matches': matches, 'structured_label_gold_score_probability_fields_selected_for_screening': False,
        'source_string_literal_classification': 'Unclassified authored source string literals; not a claim that every literal excludes labels or gold.',
        'upstream_rows_imported': 0,
        'limits': ['Visible lexical screen only; no semantic or foundation-pretraining independence proof.',
                   'Author policy/filter journals omit full inputs; 94 injection texts are truncated.',
                   'Game/browser logs expose partial states. Public JevBench is development feedback.',
                   'Partial lexical matching uses state strings only; question/options are covered by full-input exact matching.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'benchmark', 'routing-root', 'raw-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-manifests', nargs='+', required=True)
    args = parser.parse_args()
    result = screen(args.dataset, args.benchmark, args.routing_root, args.raw_dir, args.source_manifests)
    with Path(args.output).open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'candidate_rows_checked', 'matched_rows')}, indent=2))
    raise SystemExit(bool(result['matched_rows']))


if __name__ == '__main__':
    main()
