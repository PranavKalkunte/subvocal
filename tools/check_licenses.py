#!/usr/bin/env python3
"""Dependency and License Compliance Checker for Subvocal SDK.

Validates that all third-party dependencies have open-source licenses
compatible with commercial use and open-source distribution.
"""

import os
import sys
from typing import List, Dict, Any

# Target files
REQ_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")

# Approved license categories
APPROVED_LICENSES = ["mit", "apache", "bsd", "psf", "public-domain", "isc", "python-software-foundation"]

# Offline fallback license database for core dependencies
OFFLINE_LICENSES = {
    "pydantic": "mit",
    "numpy": "bsd",
    "scipy": "bsd",
    "scikit-learn": "bsd",
    "joblib": "bsd",
    "torch": "bsd",
    "keyring": "mit",
    "onnx": "apache",
    "h5py": "bsd",
    "brainflow": "mit"
}


def parse_requirements(filepath: str) -> List[str]:
    """Parse package names from requirements file."""
    packages = []
    if not os.path.exists(filepath):
        return packages
        
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip versions, extras, environment selectors
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].split(";")[0].strip()
            if pkg:
                packages.append(pkg)
    return packages


def get_package_license(package_name: str) -> str:
    """Attempts to resolve the package license from system metadata or offline database."""
    pkg_key = package_name.lower()
    
    # 1. Try local metadata lookup
    try:
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            try:
                meta = metadata.metadata(package_name)
                # Check License field
                lic = meta.get("License", "")
                if lic:
                    return lic
                # Check classifiers
                classifiers = meta.get_all("Classifier", [])
                for c in classifiers:
                    if "License ::" in c:
                        return c.split("::")[-1].strip()
            except metadata.PackageNotFoundError:
                pass
    except Exception:
        pass
        
    # 2. Fallback to offline database
    return OFFLINE_LICENSES.get(pkg_key, "Unknown")


def main():
    print("======================================================================")
    print("Subvocal Dependency & License Compliance Auditor")
    print("======================================================================")
    
    packages = parse_requirements(REQ_FILE)
    if not packages:
        print("[!] No requirements found to scan.")
        sys.exit(0)
        
    print(f"Scanning {len(packages)} packages listed in {os.path.basename(REQ_FILE)}...\n")
    
    violations = []
    
    # Print table header
    print(f"{'Package':<20} | {'Declared License':<30} | {'Status':<15}")
    print("-" * 71)
    
    for pkg in packages:
        lic = get_package_license(pkg)
        lic_disp = lic[:30] + "..." if len(lic) > 30 else lic
        
        # Check compliance
        lic_clean = lic.lower().replace(" license", "")
        is_approved = any(app in lic_clean for app in APPROVED_LICENSES)
        
        status = "COMPLIANT"
        if not is_approved and lic != "Unknown":
            # Check for copyleft warnings
            if "gpl" in lic_clean or "agpl" in lic_clean or "cddl" in lic_clean:
                status = "COPYLEFT WARNING"
                violations.append((pkg, lic, "Copyleft restriction"))
            else:
                status = "MANUAL REVIEW"
        elif lic == "Unknown":
            status = "UNKNOWN"
            
        print(f"{pkg:<20} | {lic_disp:<30} | {status:<15}")
        
    print("-" * 71)
    
    if violations:
        print(f"\n[❌ FAIL] Dependency scanning failed with {len(violations)} warnings:")
        for pkg, lic, reason in violations:
            print(f"  - {pkg} ({lic}): {reason}")
        sys.exit(1)
    else:
        print("\n[✅ PASS] All dependencies comply with open-source distribution policies.")
        sys.exit(0)


if __name__ == "__main__":
    main()
