#!/usr/bin/env python3
"""Run Script Validator

Validates run.sh scripts for:
- Shebang presence
- License header
- Error handling (set -e)
- Output format for performance metrics

SLURM batch scripts (.sbatch / .slurm) are submitted with sbatch rather than
executed, so they are not required to be executable.
"""

import os
import re
from typing import List

try:
    from .model_config_validator import _dir_emits_performance_line
except ImportError:  # run as a script: python3 tools/validation/script_validator.py
    from model_config_validator import _dir_emits_performance_line

BATCH_SUFFIXES = ('.sbatch', '.slurm')


class ScriptValidator:
    """Validator for MAD run scripts"""

    def __init__(self, verbose: bool = True, multiple_results: bool = False):
        self.verbose = verbose
        # The model writes its results to a CSV (models.json multiple_results)
        # instead of printing a 'performance:' line.
        self.multiple_results = multiple_results
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
        if not script_path.endswith(BATCH_SUFFIXES):
            self._check_executable(script_path)
        self._check_performance_output(script_path)

        if self.verbose:
            self._print_report(script_path)

        return len(self.errors) == 0

    def _check_shebang(self, content: str):
        """Check for proper shebang"""
        first_line = content.split('\n', 1)[0].strip()
        if not re.match(r'^#!\s*(/usr)?/bin/(env\s+)?(bash|sh)(\s|$)', first_line):
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

    def _check_performance_output(self, script_path: str):
        """Check the script emits the 'performance:' metric line MAD parses.

        The line may come from a helper the script calls, so the script's whole
        directory is searched. Skipped for multiple_results models.
        """
        if self.multiple_results:
            return
        if not _dir_emits_performance_line(os.path.dirname(os.path.abspath(script_path))):
            self.warnings.append(
                "No 'performance: <value> <unit>' line in the script's directory "
                "(fine only if the model sets multiple_results; pass --multiple-results)"
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
    parser.add_argument('--multiple-results', action='store_true',
                        help='The model sets multiple_results, so no performance: line is expected')

    args = parser.parse_args()

    validator = ScriptValidator(verbose=True, multiple_results=args.multiple_results)
    valid = validator.validate(args.script)

    sys.exit(0 if valid else 1)
