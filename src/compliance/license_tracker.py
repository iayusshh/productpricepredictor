"""
License tracking and validation system for ML Product Pricing Challenge.

This module ensures compliance with competition rules by tracking all dependencies,
model checkpoints, and data sources used in the solution.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pkg_resources


class LicenseTracker:
    """Tracks and validates licenses for all dependencies and model checkpoints."""
    
    ALLOWED_LICENSES = {
        'MIT License',
        'MIT',
        'Apache License 2.0',
        'Apache 2.0',
        'Apache Software License',
        'BSD License',
        'BSD',
        'BSD-3-Clause',
        'BSD-2-Clause',
        'Python Software Foundation License',
        'PSF',
        'Mozilla Public License 2.0',
        'MPL-2.0'
    }
    
    RESTRICTED_LICENSES = {
        'GPL',
        'LGPL',
        'AGPL',
        'Commercial',
        'Proprietary'
    }
    
    def __init__(self, project_root: str = "."):
        """Initialize license tracker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.compliance_log = []
        
    def get_installed_packages(self) -> Dict[str, str]:
        """Get all installed packages and their versions.
        
        Returns:
            Dictionary mapping package names to versions
        """
        packages = {}
        try:
            installed_packages = [d for d in pkg_resources.working_set]
            for package in installed_packages:
                packages[package.project_name] = package.version
        except Exception as e:
            self.logger.error(f"Error getting installed packages: {e}")
            
        return packages
    
    def get_package_license(self, package_name: str) -> Optional[str]:
        """Get license information for a specific package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            License string if found, None otherwise
        """
        try:
            # Try to get license from package metadata
            dist = pkg_resources.get_distribution(package_name)
            if dist.has_metadata('METADATA'):
                metadata = dist.get_metadata('METADATA')
                for line in metadata.split('\n'):
                    if line.startswith('License:'):
                        return line.split(':', 1)[1].strip()
            
            # Try alternative metadata formats
            if dist.has_metadata('PKG-INFO'):
                metadata = dist.get_metadata('PKG-INFO')
                for line in metadata.split('\n'):
                    if line.startswith('License:'):
                        return line.split(':', 1)[1].strip()
                        
        except Exception as e:
            self.logger.warning(f"Could not get license for {package_name}: {e}")
            
        return None
    
    def validate_license(self, license_str: str) -> Tuple[bool, str]:
        """Validate if a license is allowed.
        
        Args:
            license_str: License string to validate
            
        Returns:
            Tuple of (is_valid, status_message)
        """
        if not license_str:
            return False, "No license information found"
            
        license_str = license_str.strip()
        
        # Check for allowed licenses
        for allowed in self.ALLOWED_LICENSES:
            if allowed.lower() in license_str.lower():
                return True, f"Allowed license: {license_str}"
        
        # Check for restricted licenses
        for restricted in self.RESTRICTED_LICENSES:
            if restricted.lower() in license_str.lower():
                return False, f"Restricted license: {license_str}"
        
        # Unknown license - flag for manual review
        return False, f"Unknown license requires manual review: {license_str}"
    
    def track_all_dependencies(self) -> Dict[str, Dict]:
        """Track licenses for all project dependencies.
        
        Returns:
            Dictionary with dependency license information
        """
        dependencies = {}
        packages = self.get_installed_packages()
        
        for package_name, version in packages.items():
            license_str = self.get_package_license(package_name)
            is_valid, status = self.validate_license(license_str)
            
            dependencies[package_name] = {
                'version': version,
                'license': license_str,
                'is_valid': is_valid,
                'status': status,
                'checked_at': datetime.now().isoformat()
            }
            
            # Log compliance entry
            self.compliance_log.append({
                'type': 'dependency',
                'name': package_name,
                'version': version,
                'license': license_str,
                'is_valid': is_valid,
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
        
        return dependencies
    
    def track_model_checkpoints(self, model_paths: List[str]) -> Dict[str, Dict]:
        """Track licenses for model checkpoints.
        
        Args:
            model_paths: List of paths to model files or directories
            
        Returns:
            Dictionary with model license information
        """
        models = {}
        
        for model_path in model_paths:
            path = Path(model_path)
            if not path.exists():
                continue
                
            # Extract model information
            model_info = self._extract_model_info(path)
            license_info = self._get_model_license_info(model_info)
            
            is_valid, status = self.validate_license(license_info.get('license'))
            
            models[str(path)] = {
                'model_info': model_info,
                'license_info': license_info,
                'is_valid': is_valid,
                'status': status,
                'checked_at': datetime.now().isoformat()
            }
            
            # Log compliance entry
            self.compliance_log.append({
                'type': 'model_checkpoint',
                'path': str(path),
                'model_info': model_info,
                'license': license_info.get('license'),
                'is_valid': is_valid,
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
        
        return models
    
    def _extract_model_info(self, model_path: Path) -> Dict:
        """Extract information about a model checkpoint.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            Dictionary with model information
        """
        info = {
            'path': str(model_path),
            'size_mb': 0,
            'model_type': 'unknown',
            'framework': 'unknown'
        }
        
        try:
            if model_path.is_file():
                info['size_mb'] = model_path.stat().st_size / (1024 * 1024)
                
                # Determine model type from extension
                suffix = model_path.suffix.lower()
                if suffix in ['.pkl', '.pickle']:
                    info['framework'] = 'sklearn/pickle'
                elif suffix in ['.pt', '.pth']:
                    info['framework'] = 'pytorch'
                elif suffix in ['.h5', '.hdf5']:
                    info['framework'] = 'tensorflow/keras'
                elif suffix == '.joblib':
                    info['framework'] = 'joblib'
                    
            elif model_path.is_dir():
                # Calculate total size of directory
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                info['size_mb'] = total_size / (1024 * 1024)
                
                # Check for framework-specific files
                if any(model_path.glob('*.bin')) or any(model_path.glob('config.json')):
                    info['framework'] = 'transformers/huggingface'
                elif any(model_path.glob('saved_model.pb')):
                    info['framework'] = 'tensorflow'
                    
        except Exception as e:
            self.logger.warning(f"Error extracting model info for {model_path}: {e}")
            
        return info
    
    def _get_model_license_info(self, model_info: Dict) -> Dict:
        """Get license information for a model.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Dictionary with license information
        """
        license_info = {
            'license': None,
            'source': 'unknown',
            'notes': ''
        }
        
        # For HuggingFace models, try to read config
        if model_info.get('framework') == 'transformers/huggingface':
            config_path = Path(model_info['path']) / 'config.json'
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Check for license in config
                    if 'license' in config:
                        license_info['license'] = config['license']
                        license_info['source'] = 'model_config'
                    
                    # Check model name for known licenses
                    model_name = config.get('_name_or_path', '').lower()
                    if 'bert' in model_name or 'roberta' in model_name:
                        license_info['license'] = license_info['license'] or 'Apache 2.0'
                        license_info['notes'] = 'Common license for BERT/RoBERTa models'
                        
                except Exception as e:
                    self.logger.warning(f"Error reading model config: {e}")
        
        # Default assumptions for common model types
        framework = model_info.get('framework', '')
        if 'sklearn' in framework:
            license_info['license'] = 'BSD'
            license_info['source'] = 'framework_default'
            license_info['notes'] = 'scikit-learn models typically BSD licensed'
        elif framework in ['pytorch', 'tensorflow']:
            license_info['notes'] = 'Framework license, model license depends on training data and architecture'
            
        return license_info
    
    def check_external_data_sources(self) -> Dict[str, Dict]:
        """Check for any external data sources being used.
        
        Returns:
            Dictionary with external data source information
        """
        external_sources = {}
        
        # Check for common external data patterns in code
        code_files = list(self.project_root.rglob('*.py'))
        
        suspicious_patterns = [
            'requests.get',
            'urllib.request',
            'wget',
            'curl',
            'api.key',
            'external_data',
            'web_scraping',
            'scrape',
            'crawl'
        ]
        
        for code_file in code_files:
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                found_patterns = []
                for pattern in suspicious_patterns:
                    if pattern in content:
                        found_patterns.append(pattern)
                
                if found_patterns:
                    external_sources[str(code_file)] = {
                        'suspicious_patterns': found_patterns,
                        'requires_review': True,
                        'checked_at': datetime.now().isoformat()
                    }
                    
                    # Log compliance entry
                    self.compliance_log.append({
                        'type': 'external_data_check',
                        'file': str(code_file),
                        'patterns_found': found_patterns,
                        'requires_review': True,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error checking file {code_file}: {e}")
        
        return external_sources
    
    def generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report.
        
        Returns:
            Complete compliance report dictionary
        """
        self.logger.info("Generating compliance report...")
        
        # Track all components
        dependencies = self.track_all_dependencies()
        model_checkpoints = self.track_model_checkpoints(self._find_model_files())
        external_sources = self.check_external_data_sources()
        
        # Calculate summary statistics
        total_deps = len(dependencies)
        valid_deps = sum(1 for dep in dependencies.values() if dep['is_valid'])
        invalid_deps = total_deps - valid_deps
        
        total_models = len(model_checkpoints)
        valid_models = sum(1 for model in model_checkpoints.values() if model['is_valid'])
        invalid_models = total_models - valid_models
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_dependencies': total_deps,
                'valid_dependencies': valid_deps,
                'invalid_dependencies': invalid_deps,
                'total_models': total_models,
                'valid_models': valid_models,
                'invalid_models': invalid_models,
                'external_sources_found': len(external_sources),
                'compliance_status': 'PASS' if (invalid_deps == 0 and invalid_models == 0 and len(external_sources) == 0) else 'REQUIRES_REVIEW'
            },
            'dependencies': dependencies,
            'model_checkpoints': model_checkpoints,
            'external_data_sources': external_sources,
            'compliance_log': self.compliance_log,
            'recommendations': self._generate_recommendations(dependencies, model_checkpoints, external_sources)
        }
        
        return report
    
    def _find_model_files(self) -> List[str]:
        """Find all model files in the project.
        
        Returns:
            List of model file paths
        """
        model_extensions = ['.pkl', '.pickle', '.pt', '.pth', '.h5', '.hdf5', '.joblib']
        model_files = []
        
        # Check models directory
        models_dir = self.project_root / 'models'
        if models_dir.exists():
            for ext in model_extensions:
                model_files.extend([str(f) for f in models_dir.rglob(f'*{ext}')])
            
            # Also check for model directories (like HuggingFace models)
            for item in models_dir.iterdir():
                if item.is_dir() and any(item.glob('config.json')):
                    model_files.append(str(item))
        
        # Check other common locations
        for pattern in ['*.pkl', '*.pt', '*.h5']:
            model_files.extend([str(f) for f in self.project_root.rglob(pattern)])
        
        return list(set(model_files))  # Remove duplicates
    
    def _generate_recommendations(self, dependencies: Dict, models: Dict, external_sources: Dict) -> List[str]:
        """Generate recommendations based on compliance check results.
        
        Args:
            dependencies: Dependency license information
            models: Model license information
            external_sources: External data source information
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for invalid dependencies
        invalid_deps = [name for name, info in dependencies.items() if not info['is_valid']]
        if invalid_deps:
            recommendations.append(f"Review licenses for dependencies: {', '.join(invalid_deps[:5])}")
            if len(invalid_deps) > 5:
                recommendations.append(f"... and {len(invalid_deps) - 5} more dependencies")
        
        # Check for invalid models
        invalid_models = [path for path, info in models.items() if not info['is_valid']]
        if invalid_models:
            recommendations.append(f"Review licenses for model checkpoints: {len(invalid_models)} models need review")
        
        # Check for external data sources
        if external_sources:
            recommendations.append(f"Review {len(external_sources)} files for potential external data usage")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All components appear compliant with competition rules")
        else:
            recommendations.append("Ensure all flagged items comply with competition rules before submission")
        
        return recommendations
    
    def save_compliance_report(self, output_path: str = "compliance_report.json") -> str:
        """Save compliance report to file.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        report = self.generate_compliance_report()
        
        output_file = self.project_root / output_path
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Compliance report saved to {output_file}")
        return str(output_file)