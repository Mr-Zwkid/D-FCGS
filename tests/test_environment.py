import os
import unittest

class TestEnvironmentVariables(unittest.TestCase):
    def test_environment_variable_exists(self):
        self.assertIn('REQUIRED_ENV_VAR', os.environ)

    def test_parameter_validity(self):
        param = os.getenv('PARAMETER')
        self.assertIsNotNone(param)
        self.assertGreater(len(param), 0)

if __name__ == '__main__':
    unittest.main()