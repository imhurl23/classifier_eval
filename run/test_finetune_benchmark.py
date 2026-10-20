import unittest

from run.finetune_benchmark import overlap, text_keys, tasks


class ExposureTests(unittest.TestCase):
    def row(self, prompt='A new request', response=None, source_id='new'):
        return {'input': {'text': prompt if response is None else f'User:\n{prompt}\n\nAssistant:\n{response}',
                          'prompt': prompt, 'response': response},
                'metadata': {'source_id': source_id, 'source': 'wildguardtest'}}

    def test_prompt_seen_without_response_excludes_pair(self):
        self.assertTrue(overlap(self.row('SAME, request!', 'Different answer'), text_keys('same request'), set()))

    def test_individual_turn_seen_in_training_pair(self):
        keys = text_keys('User:\nTraining request\n\nAssistant:\nPreviously supplied answer')
        self.assertTrue(overlap(self.row(response='Previously supplied answer'), keys, set()))

    def test_source_id_excludes_changed_text(self):
        self.assertTrue(overlap(self.row(source_id='known'), set(), {'known'}))

    def test_unseen_sensitive_topic_remains(self):
        self.assertFalse(overlap(self.row('A different question about crime'), text_keys('A question about crime'), set()))

    def test_refusal_is_separate_and_no_derived_call(self):
        self.assertEqual(tasks(self.row()), ('prompt_harm', 'response_harm', 'response_refusal'))


if __name__ == '__main__':
    unittest.main()
