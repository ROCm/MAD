#!/usr/bin/env python3
"""Dockerfile Validator

Validates Dockerfiles for:
- Required CONTEXT header
- MIT license presence
- Base image specification
- Working directory setup
- Common best practices
"""

import re
from typing import List, Tuple


class DockerfileValidator:
    """Validator for MAD Dockerfiles"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, dockerfile_path: str) -> bool:
        """
        Validate a Dockerfile

        Args:
            dockerfile_path: Path to Dockerfile

        Returns:
            True if validation passes, False if errors found
        """
        self.errors = []
        self.warnings = []

        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            self.errors.append(f"Dockerfile not found: {dockerfile_path}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading Dockerfile: {e}")
            return False

        # Check required elements
        self._check_context_header(content)
        self._check_license_header(content)
        self._check_base_image(content)
        self._check_workspace_setup(content)
        self._check_best_practices(content)

        if self.verbose:
            self._print_report(dockerfile_path)

        return len(self.errors) == 0

    def _check_context_header(self, content: str):
        """Check for required CONTEXT header"""
        if "# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}" not in content:
            self.errors.append(
                "Missing required CONTEXT header. "
                "Should be: # CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}"
            )

    def _check_license_header(self, content: str):
        """Check for MIT license header"""
        if "MIT License" not in content:
            self.warnings.append("Missing MIT License header")

        if "Copyright (c)" not in content or "Advanced Micro Devices" not in content:
            self.warnings.append("Missing AMD copyright notice")

    def _check_base_image(self, content: str):
        """Check base image is properly specified"""
        if "ARG BASE_DOCKER=" not in content:
            self.warnings.append(
                "Missing ARG BASE_DOCKER. "
                "Consider using ARG for base image flexibility"
            )

        if not re.search(r'^\s*FROM\s', content, re.MULTILINE):
            self.errors.append("Missing FROM instruction")

    def _check_workspace_setup(self, content: str):
        """Check workspace directory setup"""
        if "WORKSPACE_DIR" not in content:
            self.warnings.append("WORKSPACE_DIR not defined")

        if "WORKDIR" not in content:
            self.warnings.append("WORKDIR not set")

    def _check_best_practices(self, content: str):
        """Check Dockerfile best practices"""
        # Check if pip list is recorded (for debugging)
        if "pip3 list" not in content and "pip list" not in content:
            self.warnings.append(
                "Consider adding 'RUN pip3 list' to record installed packages"
            )

        # Check for apt cache cleanup
        if "apt update" in content or "apt install" in content:
            if "rm -rf /var/lib/apt/lists/*" not in content:
                self.warnings.append(
                    "Consider cleaning apt cache with: "
                    "&& rm -rf /var/lib/apt/lists/*"
                )

        # Check for USER directive
        if "USER root" not in content:
            self.warnings.append("Consider explicitly setting USER root")

    def _print_report(self, dockerfile_path: str):
        """Print validation report"""
        print(f"\nDockerfile Validation Report: {dockerfile_path}")
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

    parser = argparse.ArgumentParser(description='Validate MAD Dockerfiles')
    parser.add_argument('dockerfile', help='Path to Dockerfile to validate')

    args = parser.parse_args()

    validator = DockerfileValidator(verbose=True)
    valid = validator.validate(args.dockerfile)

    sys.exit(0 if valid else 1)
