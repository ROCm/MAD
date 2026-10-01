#!/usr/bin/env python3
"""Run Script Validator

Validates run.sh scripts for:
- Shebang presence
- License header
- Error handling (set -e)
- Required environment variables
- Output format for performance metrics
"""

import os
from typing import List


class ScriptValidator:
    """Validator for MAD run scripts"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, script_path: str) -> bool:
        """
        Validate a run script

        Args:
            script_path: Path to run.sh script

        Returns:
            True if validation passes, False if errors found
        """
        self.errors = []
        self.warnings = []

        try:
            with open(script_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            self.errors.append(f"Script not found: {script_path}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading script: {e}")
            return False

        # Check required elements
        self._check_shebang(content)
        self._check_license_header(content)
        self._check_error_handling(content)
        self._check_executable(script_path)
        self._check_required_env_vars(content)
        self._check_performance_output(content)

        if self.verbose:
            self._print_report(script_path)

        return len(self.errors) == 0

    def _check_shebang(self, content: str):
        """Check for proper shebang"""
        first_line = content.split('\n', 1)[0].strip()
        if first_line not in ('#!/bin/bash', '#!/usr/bin/env bash'):
            self.errors.append(
                "Missing or incorrect shebang. Should be: #!/bin/bash "
                "or #!/usr/bin/env bash"
            )

    def _check_license_header(self, content: str):
        """Check for MIT license header"""
        if "MIT License" not in content:
            self.warnings.append("Missing MIT License header")

        if "Copyright" not in content:
            self.warnings.append("Missing copyright notice")

    def _check_error_handling(self, content: str):
        """Check for error handling"""
        if "set -e" not in content and "set -ex" not in content:
            self.warnings.append(
                "Consider adding 'set -e' or 'set -ex' for error handling"
            )

    def _check_executable(self, script_path: str):
        """Check if script is executable"""
        if not os.access(script_path, os.X_OK):
            self.warnings.append(
                f"Script is not executable. Run: chmod +x {script_path}"
            )

    def _check_required_env_vars(self, content: str):
        """Check for the MAD runtime environment variables the script relies on"""
        required_vars = ['MAD_MODEL_NAME', 'MAD_RUNTIME_NGPUS']
        for var in required_vars:
            if var not in content:
                self.warnings.append(
                    f"Script does not reference required MAD environment variable: {var}"
                )

    def _check_performance_output(self, content: str):
        """Check the script emits the 'performance:' metric line MAD parses"""
        if "performance:" not in content:
            self.warnings.append(
                "Script does not appear to output a 'performance: <value> <unit>' line"
            )

    def _print_report(self, script_path: str):
        """Print validation report"""
        print(f"\nScript Validation Report: {script_path}")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
        elif not self.errors:
            print(f"\n✅ No errors ({len(self.warnings)} warnings)")
        else:
            print(f"\n❌ Validation failed: {len(self.errors)} errors")

        print()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Validate MAD run scripts')
    parser.add_argument('script', help='Path to script to validate')

    args = parser.parse_args()

    validator = ScriptValidator(verbose=True)
    valid = validator.validate(args.script)

    sys.exit(0 if valid else 1)
