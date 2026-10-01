#!/usr/bin/env python3
"""Model Configuration Validator

Validates models.json entries for:
- Required fields presence
- Naming conventions
- Field types
- Cross-field consistency
- Framework-specific requirements
"""

import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

# Repo root, derived from this file's location (tools/validation/) so registry
# discovery does not depend on the caller's current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _str_field(model_entry: dict, field: str) -> str:
    """Return model_entry[field] if it is a string, else ''.

    Type errors are reported by _check_field_types; name/path-dependent checks
    use this so malformed values are skipped instead of crashing.
    """
    value = model_entry.get(field, '')
    return value if isinstance(value, str) else ''


class ModelConfigValidator:
    """Validator for MAD model configuration entries"""

    REQUIRED_FIELDS = ['name', 'dockerfile', 'scripts', 'n_gpus', 'owner', 'tags']
    OPTIONAL_FIELDS = ['url', 'data', 'timeout', 'multiple_results',
                       'training_precision', 'skip_gpu_arch']

    # Framework-specific validation rules
    FRAMEWORK_RULES = {
        'pyt_vllm': {
            'required_tags': ['pyt', 'vllm'],
            'recommended_args': ['--config', '--model_repo'],
        },
        'pyt_sglang': {
            'required_tags': ['pyt', 'sglang'],
            'recommended_args': ['--model_repo', '--test_option'],
        },
        'jax_maxtext': {
            'required_tags': ['jax'],
        },
    }

    def __init__(self, verbose: bool = True, base_dir: str = '.',
                 allow_missing_files: bool = False):
        self.verbose = verbose
        self.base_dir = base_dir
        self.allow_missing_files = allow_missing_files
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, model_entry: dict) -> bool:
        """
        Validate a model configuration entry

        Args:
            model_entry: Dictionary containing model configuration

        Returns:
            True if validation passes (warnings OK), False if errors found
        """
        self.errors = []
        self.warnings = []

        if not isinstance(model_entry, dict):
            self.errors.append(
                f"Model entry must be a JSON object, got {type(model_entry).__name__}"
            )
            if self.verbose:
                self._print_report('Unknown')
            return False

        # Level 1: Required fields
        self._check_required_fields(model_entry)

        # Level 2: Naming convention
        self._check_naming_convention(model_entry)

        # Level 3: Field types
        self._check_field_types(model_entry)

        # Level 4: Cross-field consistency
        self._check_cross_field_consistency(model_entry)

        # Level 5: Framework-specific rules
        self._check_framework_rules(model_entry)

        # Level 6: File references
        self._check_file_references(model_entry)

        if self.verbose:
            self._print_report(model_entry.get('name', 'Unknown'))

        return len(self.errors) == 0

    def _check_required_fields(self, model_entry: dict):
        """Check all required fields are present"""
        for field in self.REQUIRED_FIELDS:
            if field not in model_entry:
                self.errors.append(f"Missing required field: {field}")
            elif not model_entry[field]:
                self.errors.append(f"Required field is empty: {field}")

    def _check_naming_convention(self, model_entry: dict):
        """Validate naming follows {framework}_{project}_{workload} pattern"""
        name = model_entry.get('name', '')
        if name is None:
            return  # already reported by _check_required_fields
        if not isinstance(name, str):
            return  # reported by _check_field_types

        # Check format
        if not re.match(r'^[a-z]+_[a-z0-9_.\-]+$', name):
            self.errors.append(
                f"Invalid name format: '{name}'. "
                "Must match pattern: {framework}_{project}_{workload} "
                "(lowercase, alphanumeric, underscores, hyphens only)"
            )
            return

        # Check at least 2 components
        parts = name.split('_')
        if len(parts) < 2:
            self.errors.append(
                f"Name must have at least 2 components: {name}"
            )

    def _check_field_types(self, model_entry: dict):
        """Verify field types are correct"""
        # name, dockerfile, scripts and owner must be strings
        for field in ('name', 'dockerfile', 'scripts', 'owner'):
            value = model_entry.get(field)
            if value is not None and not isinstance(value, str):
                self.errors.append(f"Field '{field}' must be a string")

        # tags must be a list
        if 'tags' in model_entry:
            if not isinstance(model_entry['tags'], list):
                self.errors.append(f"Field 'tags' must be a list")
            elif len(model_entry['tags']) == 0:
                self.warnings.append(f"Field 'tags' is empty")

        # n_gpus must be a string (for backward compatibility)
        if 'n_gpus' in model_entry:
            if not isinstance(model_entry['n_gpus'], str):
                self.errors.append(f"Field 'n_gpus' must be a string")

        # timeout must be a number if present
        if 'timeout' in model_entry:
            if not isinstance(model_entry['timeout'], (int, float)):
                self.errors.append(f"Field 'timeout' must be a number")

        # args must be present and a string (the runtime indexes model['args']
        # unconditionally and appends it to the shell command, so a missing
        # or non-string value breaks at run time; unlike other required
        # fields, an empty string is valid here)
        if 'args' not in model_entry:
            self.errors.append(f"Missing required field: args")
        elif not isinstance(model_entry['args'], str):
            self.errors.append(f"Field 'args' must be a string")

    def _check_cross_field_consistency(self, model_entry: dict):
        """Check fields are consistent with each other"""
        name = _str_field(model_entry, 'name')
        tags = model_entry.get('tags', [])

        # Framework tag should match name prefix (skip if tags isn't a list;
        # _check_field_types already records an error for that case)
        if name and tags and isinstance(tags, list):
            framework_prefix = name.split('_')[0]
            if framework_prefix not in tags:
                self.warnings.append(
                    f"Framework tag '{framework_prefix}' not in tags list. "
                    f"Tags: {tags}"
                )

        # Dockerfile path should match naming pattern
        dockerfile = _str_field(model_entry, 'dockerfile')
        if dockerfile and name:
            expected_prefix = '_'.join(name.split('_')[:2])  # e.g., pyt_vllm
            if expected_prefix not in dockerfile:
                self.warnings.append(
                    f"Dockerfile path '{dockerfile}' doesn't match expected "
                    f"pattern 'docker/{expected_prefix}'"
                )

    def _check_framework_rules(self, model_entry: dict):
        """Apply framework-specific validation rules"""
        name = _str_field(model_entry, 'name')
        if not name:
            return

        # Determine framework key (first two components)
        framework_key = '_'.join(name.split('_')[:2])

        if framework_key not in self.FRAMEWORK_RULES:
            return  # No specific rules for this framework

        rules = self.FRAMEWORK_RULES[framework_key]
        tags = model_entry.get('tags', [])
        args = model_entry.get('args', '')

        # Check required tags (skip if tags isn't a list; _check_field_types
        # already records an error for that case)
        if 'required_tags' in rules and isinstance(tags, list):
            for req_tag in rules['required_tags']:
                if req_tag not in tags:
                    self.errors.append(
                        f"Missing required tag for {framework_key}: {req_tag}"
                    )

        # Check recommended args (run even when args is empty/omitted, so a
        # model with no args at all is still flagged for missing recommendations).
        # Skip if args isn't a string (e.g. null) rather than crashing on `in`.
        if 'recommended_args' in rules and isinstance(args, str):
            for rec_arg in rules['recommended_args']:
                if rec_arg not in args:
                    self.warnings.append(
                        f"Recommended argument missing: {rec_arg}"
                    )

        # Check required fields for framework
        if 'required_fields' in rules:
            for req_field in rules['required_fields']:
                if req_field not in model_entry or not model_entry[req_field]:
                    self.errors.append(
                        f"Required field for {framework_key}: {req_field}"
                    )

    def _check_file_references(self, model_entry: dict):
        """Verify referenced files and directories exist.

        Paths in model_entry (dockerfile, scripts) are relative to the
        directory containing the models.json file, not the process cwd.
        """
        dockerfile = _str_field(model_entry, 'dockerfile')
        if dockerfile:
            dockerfile_path = os.path.normpath(
                os.path.join(self.base_dir, f"{dockerfile}.ubuntu.amd.Dockerfile")
            )
            if not os.path.exists(dockerfile_path):
                message = f"Dockerfile not found: {dockerfile_path}"
                if self.allow_missing_files:
                    self.warnings.append(f"{message} (OK if generating a new model)")
                else:
                    self.errors.append(message)

        scripts = _str_field(model_entry, 'scripts')
        if scripts:
            scripts_path = os.path.normpath(os.path.join(self.base_dir, scripts))
            if not os.path.exists(scripts_path):
                message = f"Scripts path not found: {scripts_path}"
                if self.allow_missing_files:
                    self.warnings.append(f"{message} (OK if generating a new model)")
                else:
                    self.errors.append(message)

    def _print_report(self, model_name: str):
        """Print validation report"""
        print(f"\nModel Validation Report for: {model_name}")
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
            print("\n✅ All validation checks passed!")
        elif not self.errors:
            print(f"\n✅ No errors found ({len(self.warnings)} warnings)")
        else:
            print(f"\n❌ Validation failed: {len(self.errors)} errors, "
                  f"{len(self.warnings)} warnings")

        print()


def discover_registry_files(repo_root: str = REPO_ROOT) -> List[str]:
    """Find every models.json registry file in the repo.

    Models are registered either in the root models.json (directories with a
    get_models_json.py) or in per-directory scripts/*/models.json files.
    """
    paths = []
    root_json = os.path.join(repo_root, 'models.json')
    if os.path.exists(root_json):
        paths.append(root_json)
    paths.extend(sorted(glob.glob(os.path.join(repo_root, 'scripts', '*', 'models.json'))))
    return paths


def _display_path(path: str, repo_root: str = REPO_ROOT) -> str:
    """Show a registry path relative to the repo root when it is inside it."""
    real_path, real_root = os.path.realpath(path), os.path.realpath(repo_root)
    if os.path.commonpath([real_path, real_root]) == real_root:
        return os.path.relpath(real_path, real_root)
    return path


def _model_names(models: list) -> List[str]:
    """Return the string names of the dict entries in `models`, in order.

    Malformed entries are skipped here; ModelConfigValidator reports them.
    """
    return [m['name'] for m in models
            if isinstance(m, dict) and isinstance(m.get('name'), str) and m['name']]


def load_registry(models_json_path: str) -> Optional[list]:
    """Load a models.json file, printing an error and returning None if it
    is missing, not valid JSON, or not a top-level list."""
    try:
        with open(models_json_path, 'r') as f:
            models = json.load(f)
    except FileNotFoundError:
        print(f"Error: {models_json_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {models_json_path}: {e}")
        return None

    if not isinstance(models, list):
        print(f"Error: {models_json_path} must contain a JSON array of model "
              f"entries, got {type(models).__name__}")
        return None
    return models


def find_cross_registry_duplicates(repo_root: str = REPO_ROOT) -> Dict[str, List[str]]:
    """Return {name: [files]} for model names registered in more than one file.

    Names must be unique across the whole registry, not just within a single
    models.json, since --model/--tags lookups match the first occurrence.
    """
    name_to_files: Dict[str, List[str]] = {}
    for path in discover_registry_files(repo_root):
        try:
            with open(path, 'r') as f:
                models = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if not isinstance(models, list):
            continue
        for name in _model_names(models):
            name_to_files.setdefault(name, []).append(path)
    return {name: files for name, files in name_to_files.items() if len(files) > 1}


def check_duplicate_names(models_json_path: str, models: list,
                          repo_root: str = REPO_ROOT) -> List[str]:
    """Return error messages for any model name duplicated within `models`
    or registered in another models.json registry file elsewhere in the repo.

    Shared by both the all-model and single-model (--model) validation paths
    so a duplicate is reported consistently regardless of which is used.
    Paths are compared by realpath, so relative, absolute and symlinked
    spellings of the same file all match.
    """
    errors: List[str] = []
    seen_names: Dict[str, int] = {}
    for name in _model_names(models):
        if name in seen_names:
            errors.append(
                f"Duplicate model name: '{name}' appears more than once in {models_json_path}"
            )
        else:
            seen_names[name] = 1

    cross_duplicates = find_cross_registry_duplicates(repo_root)
    real_path = os.path.realpath(models_json_path)
    for name, files in cross_duplicates.items():
        real_files = {os.path.realpath(f) for f in files}
        if real_path in real_files and len(real_files) > 1:
            other_files = sorted(_display_path(f, repo_root)
                                 for f in real_files if f != real_path)
            errors.append(
                f"Duplicate model name across registries: '{name}' also "
                f"registered in: {', '.join(other_files)}"
            )
    return errors


def validate_models_json(models_json_path: str = "models.json",
                          allow_missing_files: bool = False) -> bool:
    """
    Validate all models in models.json

    Args:
        models_json_path: Path to models.json file
        allow_missing_files: Treat missing Dockerfile/script references as
            warnings instead of errors (for models still being generated)

    Returns:
        True if all models are valid, False otherwise
    """
    models = load_registry(models_json_path)
    if models is None:
        return False

    base_dir = os.path.dirname(models_json_path) or '.'
    validator = ModelConfigValidator(verbose=True, base_dir=base_dir,
                                      allow_missing_files=allow_missing_files)
    all_valid = True

    for model in models:
        if not validator.validate(model):
            all_valid = False

    for message in check_duplicate_names(models_json_path, models):
        print(f"\n❌ {message}")
        all_valid = False

    return all_valid


def validate_all_registries(allow_missing_files: bool = False) -> bool:
    """Validate every registry file in the repo (root models.json plus every
    scripts/*/models.json), since models are registered per-directory."""
    registry_files = discover_registry_files()
    if not registry_files:
        # Never report success without having validated anything
        print(f"Error: no models.json registry files found under {REPO_ROOT}")
        return False
    all_valid = True
    for path in registry_files:
        if not validate_models_json(path, allow_missing_files):
            all_valid = False
    return all_valid


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate MAD model configurations')
    parser.add_argument('--models-json', default='models.json',
                       help='Path to a single models.json file')
    parser.add_argument('--model', help='Validate specific model by name')
    parser.add_argument('--all', action='store_true',
                       help='Validate every registry file in the repo (root '
                            'models.json plus every scripts/*/models.json), '
                            'instead of just --models-json')
    parser.add_argument('--allow-missing-files', action='store_true',
                       help='Treat missing Dockerfile/script references as '
                            'warnings instead of errors (for models still '
                            'being generated)')

    args = parser.parse_args()

    if args.model:
        # Validate single model
        try:
            models = load_registry(args.models_json)
            if models is None:
                sys.exit(1)

            model_entry = None
            for m in models:
                if isinstance(m, dict) and m.get('name') == args.model:
                    model_entry = m
                    break

            if not model_entry:
                print(f"Error: Model '{args.model}' not found in {args.models_json}")
                sys.exit(1)

            base_dir = os.path.dirname(args.models_json) or '.'
            validator = ModelConfigValidator(verbose=True, base_dir=base_dir,
                                              allow_missing_files=args.allow_missing_files)
            valid = validator.validate(model_entry)

            for message in check_duplicate_names(args.models_json, models):
                if f"'{args.model}'" in message:
                    print(f"\n❌ {message}")
                    valid = False

            sys.exit(0 if valid else 1)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.all:
        # Validate every registry file in the repo
        valid = validate_all_registries(args.allow_missing_files)
        sys.exit(0 if valid else 1)
    else:
        # Validate a single models.json file
        valid = validate_models_json(args.models_json, args.allow_missing_files)
        sys.exit(0 if valid else 1)
