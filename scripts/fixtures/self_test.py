"""FAKE fixtures and offline regression tests. All generated files stay here."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_watchdogs as runner
import score
import check_leakage as leakage


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        if path.suffix == '.jsonl':
            for row in obj:
                f.write(json.dumps(row) + '\n')
        else:
            json.dump(obj, f)


class OfflineTests(unittest.TestCase):
    def test_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'fixtures') as temp:
            d = Path(temp)
            for condition in ('neutral', 'plain', 'evidence'):
                args = ['--packets', str(ROOT/'fixtures/packets.jsonl'), '--prompt', str(ROOT/'fixtures/prompt.txt'),
                        '--models', 'fake-A,fake-B', '--condition', condition, '--out-dir', str(d/'runs'),
                        '--mock', str(ROOT/'fixtures/responses.json'), '--prices', str(ROOT/'fixtures/prices.json')]
                with patch.object(runner, 'request_response', side_effect=AssertionError('Network forbidden')), patch.object(runner, 'read_key', side_effect=AssertionError('Credentials forbidden')):
                    self.assertEqual(runner.main(args), 0)
                    with self.assertRaises(FileExistsError):
                        runner.main(args)
            self.assertEqual(score.main(['--verdicts', str(d/'runs'), '--gold', str(ROOT/'fixtures/packets.jsonl'), '--out-dir', str(d/'scores')]), 0)
            with (d/'scores/scores.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            overall = [r for r in rows if r['dimension'] == 'overall']
            self.assertEqual(len(overall), 6)
            for r in overall:
                self.assertEqual(int(r['n']), 5)
                self.assertEqual(int(r['laundered_misses']), int(r['model'] == 'fake-B'))
                self.assertEqual(int(r['supported_false_alarms']), int(r['model'] == 'fake-B'))
                self.assertEqual(int(r['benign_false_alarms']), int(r['model'] == 'fake-B'))
                self.assertAlmostEqual(float(r['cost_per_1000_usd']), 0.02)
            self.assertEqual({r['value'] for r in rows if r['dimension']=='record_type'}, {'ticket','review_log','minutes','changelog','email'})
            duplicate = d/'duplicate.jsonl'
            duplicate.write_text(next((d/'runs').glob('*.jsonl')).read_text())
            with self.assertRaises(ValueError):
                score.main(['--verdicts', str(d/'runs'), str(duplicate), '--gold', str(ROOT/'fixtures/packets.jsonl'), '--out-dir', str(d/'bad')])

    def test_invalid_missing_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'fixtures') as temp:
            d = Path(temp)
            mock = json.loads((ROOT/'fixtures/responses.json').read_text())
            mock['neutral']['fake-A']['fake-1'] = {'output_text':'invalid JSON'}
            write(d/'bad.json', mock)
            args = ['--packets', str(ROOT/'fixtures/packets.jsonl'), '--prompt', str(ROOT/'fixtures/prompt.txt'),
                    '--models', 'fake-A', '--condition', 'neutral', '--out-dir', str(d/'run'), '--mock', str(d/'bad.json')]
            self.assertEqual(runner.main(args), 1)
            score_args = ['--verdicts', str(d/'run'), '--gold', str(ROOT/'fixtures/packets.jsonl'), '--out-dir', str(d/'scores')]
            self.assertEqual(score.main(score_args), 0)
            with (d/'scores/scores.csv').open() as handle:
                overall = next(r for r in csv.DictReader(handle) if r['dimension']=='overall')
            self.assertEqual(overall['invalid'], '1')
            self.assertEqual(overall['cost_per_1000_usd'], '')
            with self.assertRaises(FileExistsError):
                score.main(score_args)
            result_path = next((d/'run').glob('*.jsonl'))
            result_rows = runner.read_rows(result_path)
            write(d/'missing.jsonl', result_rows[:-1])
            with self.assertRaises(ValueError):
                score.main(['--verdicts', str(d/'missing.jsonl'), '--gold', str(ROOT/'fixtures/packets.jsonl'), '--out-dir', str(d/'missing-score')])

    def test_leakage(self):
        f = ROOT/'fixtures'
        base = ['--heldout', str(f/'heldout.jsonl'), '--conartist', str(f/'conartist.jsonl'), '--heldout-families', 'reserved', '--heldout-record-types', 'email']
        self.assertEqual(leakage.main(['--train', str(f/'train_clean.jsonl')] + base), 0)
        self.assertEqual(leakage.main(['--train', str(f/'train_leaky.jsonl')] + base), 1)
        with tempfile.TemporaryDirectory(dir=f) as temp:
            for row, reason in [({'id':'held-1','packet_text':'unrelated orange kite'}, 'case_id'),
                                ({'id':'new','packet_text':'THE protected, passage has several distinct words!'}, 'normalized_exact'),
                                ({'id':'new','packet_text':'the protected passage has several distinct words appended'}, 'ngram_overlap'),
                                ({'id':'new','packet_text':'unrelated orange kite','family':'reserved'}, 'heldout_family'),
                                ({'id':'new','packet_text':'unrelated orange kite','record_type':'email'}, 'heldout_record_type'),
                                ({'id':'con-1','packet_text':'unrelated orange kite'}, 'case_id')]:
                path = Path(temp)/(reason + '-' + row['id'] + '.jsonl')
                write(path, [row])
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(leakage.main(['--train', str(path)] + base), 1)
                self.assertIn(reason, output.getvalue())

    def test_parser_and_redaction(self):
        self.assertEqual(runner.parse_verdict('```json\n{"verdict":"flag","evidence":"fake"}\n```')['verdict'], 'flag')
        for text in ('no JSON', '{"verdict":"maybe","evidence":"x"}', '{"verdict":"clear"}'):
            with self.assertRaises(ValueError):
                runner.parse_verdict(text)
        self.assertNotIn('SYNTHETIC_SECRET', json.dumps(runner.redact({'SYNTHETIC_SECRET':['aSYNTHETIC_SECRETb']}, 'SYNTHETIC_SECRET')))
        with tempfile.TemporaryDirectory(dir=ROOT/'fixtures') as temp:
            path = Path(temp)/'fake.env'
            path.write_text('# FAKE\nADAPTION_API_KEY="SYNTHETIC_SECRET"\n')
            self.assertEqual(runner.read_key(path), 'SYNTHETIC_SECRET')

    def test_retry_transport(self):
        # Patch the module entirely: requests need not be installed for offline tests.
        from types import SimpleNamespace
        class TransportError(Exception):
            pass
        ok = SimpleNamespace(status_code=200, json=lambda: {'output_text':'OK'})
        fake = SimpleNamespace(RequestException=TransportError)
        from unittest.mock import Mock
        fake.post = Mock(side_effect=[TransportError(), SimpleNamespace(status_code=429), ok])
        with patch.dict(sys.modules, {'requests':fake}), patch.object(runner.time, 'sleep'):
            data, error, attempts = runner.request_response('https://fake.invalid/v1', 'SYNTHETIC', {}, 2, 1)
        self.assertEqual((error, attempts), (None, 3))
        self.assertEqual(data['output_text'], 'OK')
        self.assertFalse(fake.post.call_args.kwargs['allow_redirects'])

    def test_syntax(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main()
