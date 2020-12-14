import json
import unittest

from science.utils import FEATURE_SYNONYM_DICT
from science.analysis.data_management import _replace_feature_words, switch_reward_function


class MyTestCase(unittest.TestCase):

    def test_replace_feature_words(self):

        base_chat = "yellow pink blue"

        # We should only replace yellow --> blue
        new_chat = _replace_feature_words(base_chat, "m_c_y", "m_c_c")
        self.assertTrue(new_chat.split()[0] in FEATURE_SYNONYM_DICT["blue"])
        self.assertTrue(new_chat.split()[1] in FEATURE_SYNONYM_DICT["pink"])
        self.assertTrue(new_chat.split()[2] in FEATURE_SYNONYM_DICT["blue"])

        base_chat = "square circle box cheesecake triangle"

        new_chat = _replace_feature_words(base_chat, "s_o_^", "^_o_s")
        self.assertTrue(new_chat.split()[0] in FEATURE_SYNONYM_DICT["triangle"])
        self.assertTrue(new_chat.split()[1] in FEATURE_SYNONYM_DICT["circle"])
        self.assertTrue(new_chat.split()[2] in FEATURE_SYNONYM_DICT["triangle"])
        self.assertTrue(new_chat.split()[3] == "cheesecake")
        self.assertTrue(new_chat.split()[4] in FEATURE_SYNONYM_DICT["square"])


if __name__ == '__main__':
    unittest.main()




