#!/usr/bin/env python3
"""
trim_requirements.py

A tool to strip out packages from a requirements.txt file (pip freeze output)
that were not picked up or used by a PyInstaller build.
"""

import argparse
import ast
import glob
import os
import re
import sys

# Default list of packages to always keep in the output requirements.
ALWAYS_KEEP = {"pyinstaller", "pip", "setuptools", "wheel"}

# Static mapping for common package-to-import name mismatches.
# Used when running outside the virtual environment (static fallback mode)
# or for packages without metadata.
STATIC_PACKAGE_TO_IMPORTS = {
    "beautifulsoup4": {"bs4"},
    "opencv-python": {"cv2"},
    "opencv-python-headless": {"cv2"},
    "pillow": {"pil"},
    "scikit-learn": {"sklearn"},
    "scikit-image": {"skimage"},
    "pyyaml": {"yaml"},
    "python-dateutil": {"dateutil"},
    "pyserial": {"serial"},
    "pyusb": {"usb"},
    "pyopenssl": {"openssl"},
    "django-debug-toolbar": {"debug_toolbar"},
    "more-itertools": {"more_itertools"},
    "ruamel.yaml": {"ruamel"},
    "pyqt5": {"pyqt5"},
    "pyqt6": {"pyqt6"},
    "pyside2": {"pyside2"},
    "pyside6": {"pyside6"},
    "protobuf": {"google.protobuf", "google"},
    "google-api-python-client": {"googleapiclient"},
    "google-auth": {"google"},
    "google-cloud-storage": {"google"},
    "dnspython": {"dns"},
    "websocket-client": {"websocket"},
    "apache-airflow": {"airflow"},
    "attrs": {"attr"},
    "flask-sqlalchemy": {"flask_sqlalchemy"},
    "flask-login": {"flask_login"},
    "flask-cors": {"flask_cors"},
    "moto": {"moto"},
    "pymongo": {"bson", "gridfs", "pymongo"},
    "pydantic-core": {"pydantic_core"},
    "bcrypt": {"bcrypt"},
    "cryptography": {"cryptography"},
    "greenlet": {"greenlet"},
    "msgpack": {"msgpack"},
    "pytest": {"pytest"},
    "sphinx": {"sphinx"},
    "jinja2": {"jinja2"},
    "markupsafe": {"markupsafe"},
    "setuptools": {"setuptools", "pkg_resources"},
    "pip": {"pip"},
    "wheel": {"wheel"},
    "pyinstaller": {"pyinstaller"},
}

# Invert static package-to-imports mapping for easy lookup.
STATIC_IMPORT_TO_PACKAGES = {}
for pkg, imps in STATIC_PACKAGE_TO_IMPORTS.items():
    for imp in imps:
        STATIC_IMPORT_TO_PACKAGES.setdefault(imp, set()).add(pkg)


def parse_requirement_line(line):
    """
    Parses a single line from a requirements file to extract the canonical package name.

    Returns (package_name, original_line) if it's a valid requirement,
    or (None, original_line) if it's a comment, empty line, or option.
    """
    stripped_line = line.strip()
    if not stripped_line or stripped_line.startswith("#"):
        return None, line

    # Handle inline comments
    clean_line = stripped_line
    if " #" in clean_line:
        clean_line = clean_line.split(" #", 1)[0].strip()

    # Handle egg parameter in URLs/editable installs (e.g., git+https://...#egg=pkg-name)
    egg_match = re.search(r"#egg=([a-zA-Z0-9._-]+)", clean_line)
    if egg_match:
        return egg_match.group(1), line

    # Handle ' @ ' syntax (PEP 508 direct references)
    if " @ " in clean_line:
        parts = clean_line.split(" @ ", 1)
        pkg_name = parts[0].strip()
        # Clean up any options like '--editable' if they precede
        pkg_name = pkg_name.split()[-1]
        return pkg_name, line

    # Strip environment markers (starting with ';')
    spec = clean_line
    if ";" in clean_line:
        spec = clean_line.split(";", 1)[0].strip()

    # Match package name at the start of the specification (before ==, >=, etc.)
    # Package name must start with an alphanumeric character per PEP 508.
    match = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)", spec)
    if match:
        return match.group(1), line

    return None, line


def get_env_package_files():
    """
    Scans the current Python environment using importlib.metadata to extract
    the files and top-level modules associated with each installed package.
    """
    pkg_to_files = {}
    pkg_to_imports = {}
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            name = dist.metadata["Name"]
            if not name:
                continue
            name_lower = name.lower()

            # Retrieve files belonging to the package
            try:
                dist_files = dist.files
                if dist_files:
                    rel_files = []
                    for file_path in dist_files:
                        # Normalize path separators to forward slashes
                        rel_path = str(file_path).replace("\\", "/")
                        rel_files.append(rel_path)
                    pkg_to_files[name_lower] = rel_files
            except Exception:
                pass

            # Retrieve top-level module names from top_level.txt metadata
            try:
                top_level = dist.read_text("top_level.txt")
                if top_level:
                    imports = [
                        line.strip().lower()
                        for line in top_level.splitlines()
                        if line.strip()
                    ]
                    if imports:
                        pkg_to_imports[name_lower] = imports
            except Exception:
                pass
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to read environment metadata: {e}\n")

    return pkg_to_files, pkg_to_imports


def get_env_mappings():
    """
    Retrieves dynamic package-to-import mappings from the running Python environment.
    """
    pkg_to_files, pkg_to_imports = get_env_package_files()

    # Also build the inverse map: import_name -> package_names
    import_to_pkgs = {}

    # packages_distributions is available in Python 3.10+
    if sys.version_info >= (3, 10):
        try:
            from importlib.metadata import packages_distributions

            for imp, pkgs in packages_distributions().items():
                imp_lower = imp.lower()
                for pkg in pkgs:
                    pkg_lower = pkg.lower()
                    import_to_pkgs.setdefault(imp_lower, set()).add(pkg_lower)
                    pkg_to_imports.setdefault(pkg_lower, []).append(imp_lower)
        except Exception:
            pass

    return pkg_to_files, pkg_to_imports, import_to_pkgs


def get_relative_to_site_packages(path_str):
    """
    Extracts the path portion relative to site-packages or dist-packages.
    Example: 'C:/venv/lib/site-packages/yaml/__init__.py' -> 'yaml/__init__.py'
    """
    path_str = path_str.replace("\\", "/")
    match = re.search(r"(?:site-packages|dist-packages)/(.*)", path_str, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def parse_toc_file(filepath):
    """
    Parses a PyInstaller TOC (Table of Contents) file which is a python script
    containing a list of tuples: (dest_name, src_name, typecode).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Locate the outer list boundaries
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1:
        try:
            toc_data = ast.literal_eval(content[start : end + 1])
            return toc_data
        except Exception as e:
            sys.stderr.write(
                f"Warning: Failed to parse TOC file with ast: {e}. Falling back to regex.\n"
            )

    # Regex fallback if AST evaluation fails
    entries = []
    pattern = r'\(\s*[\'"]([^\'"]+)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
    for match in re.finditer(pattern, content):
        entries.append((match.group(1), match.group(2), match.group(3)))
    return entries


def parse_build_log(log_content):
    """
    Parses the plain text PyInstaller build log to extract referenced file paths
    and imported module names.
    """
    referenced_files = set()
    imported_modules = set()

    # 1. Extract paths referencing site-packages / dist-packages
    path_pattern = r"(?:site-packages|dist-packages)[/\\]([^\'\"\s,;]+)"
    for match in re.finditer(path_pattern, log_content, re.IGNORECASE):
        rel_path = match.group(1).replace("\\", "/")
        referenced_files.add(rel_path)

    # 2. Extract modules loaded via hooks
    hook_pattern = (
        r"(?:Loading|Processing) module hook [\'\"]hook-([a-zA-Z0-9._-]+)\.py[\'\"]"
    )
    for match in re.finditer(hook_pattern, log_content):
        imported_modules.add(match.group(1).lower())

    # 3. Extract module graph cashing statements
    graph_pattern = r"Caching module graph member\s+([a-zA-Z0-9._]+)"
    for match in re.finditer(graph_pattern, log_content):
        top_level = match.group(1).split(".")[0].lower()
        imported_modules.add(top_level)

    # 4. Extract explicit import debug statements (if --debug=imports is used)
    debug_patterns = [
        r"\bimport\s+[\'\"]([a-zA-Z0-9._]+)[\'\"]",
        r"\bimporting\s+([a-zA-Z0-9._]+)",
        r"\bimport\s+([a-zA-Z0-9._]+)",
    ]
    for pattern in debug_patterns:
        for match in re.finditer(pattern, log_content):
            top_level = match.group(1).split(".")[0].lower()
            imported_modules.add(top_level)

    return referenced_files, imported_modules


def find_toc_files_in_dir(build_dir):
    """
    Searches a directory recursively for PyInstaller TOC files.
    """
    toc_files = glob.glob(os.path.join(build_dir, "Analysis-*.toc"))
    if not toc_files:
        toc_files = glob.glob(
            os.path.join(build_dir, "**/Analysis-*.toc"), recursive=True
        )
    if not toc_files:
        toc_files = glob.glob(os.path.join(build_dir, "*.toc"))
    return toc_files


def read_pyinstaller_data(args):
    """
    Finds and reads PyInstaller build details based on CLI arguments.
    Returns sets of (referenced_files, imported_modules).
    """
    # 1. Read from build directory if specified
    if args.build_dir:
        toc_files = find_toc_files_in_dir(args.build_dir)
        if not toc_files:
            raise FileNotFoundError(
                f"No TOC files found in build directory: {args.build_dir}"
            )
        return parse_tocs(toc_files, args.verbose)

    # 2. Read from TOC file directly if specified
    if args.toc:
        return parse_tocs([args.toc], args.verbose)

    # 3. Read from log file/stdin
    if args.log:
        log_content = ""
        if args.log == "-":
            if args.verbose:
                print("Reading build log from stdin...", file=sys.stderr)
            log_content = sys.stdin.read()
        else:
            if args.verbose:
                print(f"Reading build log file: {args.log}", file=sys.stderr)
            with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()
        return parse_build_log(log_content)

    # 4. Auto-detect build folder in current directory
    if os.path.isdir("build"):
        toc_files = find_toc_files_in_dir("build")
        if toc_files:
            if args.verbose:
                print(
                    f"Auto-detected TOC files in build folder: {toc_files}",
                    file=sys.stderr,
                )
            return parse_tocs(toc_files, args.verbose)

    raise ValueError(
        "No PyInstaller input specified. Provide a log file (-l), TOC file (-t), "
        "build folder (-b), or run this script where a 'build' folder exists."
    )


def parse_tocs(toc_paths, verbose=False):
    """
    Helper to extract referenced files and imported modules from a list of TOC paths.
    """
    referenced_files = set()
    imported_modules = set()
    for toc_path in toc_paths:
        if verbose:
            print(f"Parsing TOC file: {toc_path}", file=sys.stderr)
        entries = parse_toc_file(toc_path)
        for dest, src, typecode in entries:
            # If it's a module, save its name
            if typecode in ("PYMODULE", "EXTENSION"):
                imported_modules.add(dest.split(".")[0].lower())
            # Record the relative file path under site-packages
            if src and os.path.isabs(src):
                rel = get_relative_to_site_packages(src)
                if rel:
                    referenced_files.add(rel)
    return referenced_files, imported_modules


def check_package_used(
    pkg_name,
    pyi_files_norm,
    pyi_imports_norm,
    pkg_to_files,
    pkg_to_imports,
    import_to_pkgs,
):
    """
    Determines if a package is used by checking its files and imports against
    what was captured during the PyInstaller build.
    """
    p_norm = pkg_name.lower()

    # 1. Always keep essential packaging dependencies
    if p_norm in ALWAYS_KEEP:
        return True, "always-keep list"

    # 2. Dynamic Environment Mode matching
    if pkg_to_files:
        p_files = {f.lower().strip("/") for f in pkg_to_files.get(p_norm, [])}
        if p_files & pyi_files_norm:
            return True, "matched via environment files copied"

    if pkg_to_imports:
        p_imports = {m.lower() for m in pkg_to_imports.get(p_norm, [])}
        if p_imports & pyi_imports_norm:
            return (
                True,
                f"matched via environment imports: {p_imports & pyi_imports_norm}",
            )

    if import_to_pkgs:
        for imp in pyi_imports_norm:
            if imp in import_to_pkgs and p_norm in import_to_pkgs[imp]:
                return (
                    True,
                    f"matched via environment import-to-package mapping for '{imp}'",
                )

    # 3. Static Fallback matching
    normalized_names = {
        p_norm,
        p_norm.replace("-", "_"),
        p_norm.replace("_", "-"),
    }

    # Static import mappings list
    static_imports = STATIC_PACKAGE_TO_IMPORTS.get(p_norm, set())
    if static_imports & pyi_imports_norm:
        return (
            True,
            f"matched via static imports mapping: {static_imports & pyi_imports_norm}",
        )

    # Direct match on normalized name in imports
    if normalized_names & pyi_imports_norm:
        return (
            True,
            f"matched via name normalization in imports: {normalized_names & pyi_imports_norm}",
        )

    # Match if any folder name in PyInstaller's files matches the package name
    for path in pyi_files_norm:
        parts = path.split("/")
        if parts:
            first_segment = parts[0]
            if first_segment in normalized_names:
                return True, f"matched folder name in path: '{first_segment}'"

    # Namespace path heuristic (e.g., google-cloud-storage -> google/cloud/storage)
    for n in normalized_names:
        ns_path = n.replace("-", "/").replace("_", "/")
        for ref_path in pyi_files_norm:
            if ref_path.startswith(ns_path + "/"):
                return True, f"matched namespace path heuristic: '{ns_path}'"

    return False, "no match found"


def get_default_output_path(req_path):
    """
    Computes a default output path by appending '-trimmed' to the input filename.
    Example: 'requirements.txt' -> 'requirements-trimmed.txt'
    """
    dirname = os.path.dirname(req_path)
    basename = os.path.basename(req_path)
    root, ext = os.path.splitext(basename)
    new_basename = f"{root}-trimmed{ext}"
    res = os.path.join(dirname, new_basename) if dirname else new_basename
    return os.path.normpath(res)


def main():
    parser = argparse.ArgumentParser(
        description="Trim requirements.txt to only include packages used in a PyInstaller build."
    )
    parser.add_argument(
        "-r",
        "--requirements",
        default="requirements.txt",
        help="Path to requirements.txt (pip freeze output). Defaults to 'requirements.txt'.",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="Path to PyInstaller build log file. Use '-' for stdin.",
    )
    parser.add_argument(
        "-b",
        "--build-dir",
        help="Path to the PyInstaller build directory containing Analysis-*.toc files.",
    )
    parser.add_argument(
        "-t",
        "--toc",
        help="Path to Analysis-*.toc file directly.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the trimmed requirements file. Defaults to '<requirements_name>-trimmed.txt'.",
    )
    parser.add_argument(
        "-k",
        "--keep",
        help="Comma-separated list of additional package names to always keep.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Force output to stdout instead of writing to a file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    # Load extra packages to keep
    if args.keep:
        extra_keep = {
            name.strip().lower() for name in args.keep.split(",") if name.strip()
        }
        ALWAYS_KEEP.update(extra_keep)

    # 1. Parse requirements file
    if not os.path.isfile(args.requirements):
        print(
            f"Error: Requirements file not found: {args.requirements}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.requirements, "r", encoding="utf-8") as f:
        req_lines = f.readlines()

    # Extract all requirements and keep index positions for formatting
    requirements_to_check = []  # List of tuples: (index, pkg_name)
    for idx, line in enumerate(req_lines):
        pkg_name, _ = parse_requirement_line(line)
        if pkg_name:
            requirements_to_check.append((idx, pkg_name))

    if not requirements_to_check:
        print(
            f"Warning: No valid package requirements found in {args.requirements}.",
            file=sys.stderr,
        )
        sys.exit(0)

    # 2. Extract PyInstaller build log/TOC data
    try:
        referenced_files, imported_modules = read_pyinstaller_data(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize referenced files for fast matching
    pyi_files_norm = {f.lower().strip("/") for f in referenced_files}
    pyi_imports_norm = {m.lower() for m in imported_modules}

    if args.verbose:
        print(
            f"Parsed {len(pyi_files_norm)} files and {len(pyi_imports_norm)} modules from PyInstaller.",
            file=sys.stderr,
        )

    # 3. Get environment package mapping (for Dynamic Environment Mode)
    pkg_to_files, pkg_to_imports, import_to_pkgs = get_env_mappings()
    has_env = bool(pkg_to_files or pkg_to_imports)
    if args.verbose:
        if has_env:
            print(
                "Running in Dynamic Environment Mode (loaded metadata from active environment).",
                file=sys.stderr,
            )
        else:
            print(
                "Running in Static Fallback Mode (active environment empty or failed to load).",
                file=sys.stderr,
            )

    # 4. Filter requirements
    kept_lines = []
    removed_packages = []

    # Map indexes of parsed lines to keep or discard
    lines_to_keep = set()

    for idx, pkg_name in requirements_to_check:
        used, reason = check_package_used(
            pkg_name,
            pyi_files_norm,
            pyi_imports_norm,
            pkg_to_files,
            pkg_to_imports,
            import_to_pkgs,
        )
        if used:
            lines_to_keep.add(idx)
            if args.verbose:
                print(f"KEPT: '{pkg_name}' ({reason})", file=sys.stderr)
        else:
            removed_packages.append(pkg_name)
            if args.verbose:
                print(f"STRIPPED: '{pkg_name}'", file=sys.stderr)

    # Reconstruct requirements content (preserving comments and structure)
    final_output = []
    for idx, line in enumerate(req_lines):
        # If line was a requirement, check if it was kept
        is_req = any(ridx == idx for ridx, _ in requirements_to_check)
        if is_req:
            if idx in lines_to_keep:
                final_output.append(line)
        else:
            # Preserve comments and empty lines
            final_output.append(line)

    # Remove extra empty lines if any were consecutive due to stripping
    cleaned_output = []
    prev_empty = False
    for line in final_output:
        is_empty = not line.strip()
        if is_empty:
            if not prev_empty:
                cleaned_output.append(line)
                prev_empty = True
        else:
            cleaned_output.append(line)
            prev_empty = False

    output_content = "".join(cleaned_output)

    # 5. Output results
    if args.stdout:
        sys.stdout.write(output_content)
    else:
        output_file = (
            args.output if args.output else get_default_output_path(args.requirements)
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_content)

        print(
            f"Success! Trimmed requirements written to: {output_file}", file=sys.stderr
        )
        print(
            f"Details: Kept {len(requirements_to_check) - len(removed_packages)}/"
            f"{len(requirements_to_check)} packages.",
            file=sys.stderr,
        )
        if removed_packages and args.verbose:
            print(f"Stripped packages: {', '.join(removed_packages)}", file=sys.stderr)


if __name__ == "__main__":
    main()
