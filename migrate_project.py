"""
Project Migration and Reorganization Script

This script helps migrate an existing disorganized RL project to the new structure.
It will:
1. Backup the current project
2. Analyze existing files
3. Reorganize into the new structure
4. Update imports
5. Generate migration report

Usage:
    python migrate_project.py --source /path/to/old/project --target /path/to/new/project
"""

import argparse
import shutil
import os
import re
from pathlib import Path
from datetime import datetime
import json


class ProjectMigrator:
    """Handles project migration and reorganization"""
    
    def __init__(self, source_dir: str, target_dir: str, backup: bool = True):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.backup_dir = None
        self.should_backup = backup
        
        self.migration_report = {
            'timestamp': datetime.now().isoformat(),
            'source': str(self.source_dir),
            'target': str(self.target_dir),
            'files_analyzed': [],
            'files_moved': [],
            'files_merged': [],
            'files_deleted': [],
            'imports_updated': [],
            'errors': []
        }
        
        # File type mappings
        self.file_categories = {
            'agents': ['*agent*.py', '*dqn*.py', '*q_learning*.py', '*policy*.py'],
            'environment': ['*env*.py', '*environment*.py', '*world*.py', '*obstacle*.py', '*sensor*.py'],
            'training': ['*train*.py', '*trainer*.py', '*learn*.py'],
            'visualization': ['*render*.py', '*visual*.py', '*plot*.py', '*gui*.py', '*display*.py'],
            'utils': ['*buffer*.py', '*replay*.py', '*logger*.py', '*metric*.py', '*util*.py', '*helper*.py']
        }
    
    def analyze_file(self, filepath: Path) -> dict:
        """Analyze a Python file to categorize it"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'path': str(filepath),
                'category': None,
                'has_class': bool(re.search(r'class\s+\w+', content)),
                'has_imports': bool(re.search(r'^import\s+|^from\s+', content, re.MULTILINE)),
                'is_test': 'test' in filepath.name.lower(),
                'is_notebook': filepath.suffix == '.ipynb',
                'line_count': len(content.split('\n')),
                'classes': re.findall(r'class\s+(\w+)', content),
                'functions': re.findall(r'def\s+(\w+)', content)
            }
            
            # Categorize based on content
            content_lower = content.lower()
            if 'agent' in content_lower or 'dqn' in content_lower or 'policy' in content_lower:
                analysis['category'] = 'agents'
            elif 'environment' in content_lower or 'env' in content_lower or 'obstacle' in content_lower:
                analysis['category'] = 'environment'
            elif 'train' in content_lower or 'trainer' in content_lower:
                analysis['category'] = 'training'
            elif 'render' in content_lower or 'visual' in content_lower or 'plot' in content_lower:
                analysis['category'] = 'visualization'
            elif 'buffer' in content_lower or 'logger' in content_lower or 'util' in content_lower:
                analysis['category'] = 'utils'
            
            return analysis
            
        except Exception as e:
            self.migration_report['errors'].append({
                'file': str(filepath),
                'error': str(e)
            })
            return None
    
    def backup_project(self):
        """Create backup of source project"""
        if not self.should_backup:
            print("Skipping backup (--no-backup flag)")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.source_dir.parent / f"backup_{self.source_dir.name}_{timestamp}"
        
        print(f"Creating backup at {self.backup_dir}...")
        shutil.copytree(self.source_dir, self.backup_dir)
        print(f"✓ Backup created successfully")
    
    def analyze_project(self):
        """Analyze all files in source project"""
        print("\nAnalyzing project structure...")
        
        py_files = list(self.source_dir.rglob("*.py"))
        nb_files = list(self.source_dir.rglob("*.ipynb"))
        
        all_files = py_files + nb_files
        print(f"Found {len(py_files)} Python files and {len(nb_files)} notebooks")
        
        for filepath in all_files:
            if filepath.suffix == '.py':
                analysis = self.analyze_file(filepath)
                if analysis:
                    self.migration_report['files_analyzed'].append(analysis)
        
        # Print summary
        categories = {}
        for file_info in self.migration_report['files_analyzed']:
            cat = file_info.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nFile categorization:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} files")
    
    def move_file(self, source_path: Path, target_path: Path):
        """Move file and update imports"""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, target_path)
        
        # Update imports if Python file
        if target_path.suffix == '.py':
            self.update_imports(target_path)
        
        self.migration_report['files_moved'].append({
            'from': str(source_path),
            'to': str(target_path)
        })
    
    def update_imports(self, filepath: Path):
        """Update import statements in a Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Common import patterns to update
            replacements = [
                (r'from\s+agents\s+import', 'from src.agents import'),
                (r'from\s+environment\s+import', 'from src.environment import'),
                (r'from\s+training\s+import', 'from src.training import'),
                (r'from\s+utils\s+import', 'from src.utils import'),
                (r'import\s+agents\.', 'import src.agents.'),
                (r'import\s+environment\.', 'import src.environment.'),
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.migration_report['imports_updated'].append(str(filepath))
        
        except Exception as e:
            self.migration_report['errors'].append({
                'file': str(filepath),
                'error': f"Import update failed: {str(e)}"
            })
    
    def reorganize_files(self):
        """Reorganize files into new structure"""
        print("\nReorganizing files...")
        
        for file_info in self.migration_report['files_analyzed']:
            source_path = Path(file_info['path'])
            category = file_info.get('category')
            
            if not category or category == 'unknown':
                print(f"  ⚠ Skipping uncategorized file: {source_path.name}")
                continue
            
            # Determine target path
            if file_info['is_test']:
                target_path = self.target_dir / 'tests' / source_path.name
            else:
                target_path = self.target_dir / 'src' / category / source_path.name
            
            # Move file
            self.move_file(source_path, target_path)
            print(f"  ✓ Moved {source_path.name} → {category}/")
        
        # Handle notebooks
        nb_files = list(self.source_dir.rglob("*.ipynb"))
        for nb_file in nb_files:
            if '.ipynb_checkpoints' not in str(nb_file):
                target_path = self.target_dir / 'notebooks' / nb_file.name
                self.move_file(nb_file, target_path)
                print(f"  ✓ Moved {nb_file.name} → notebooks/")
    
    def identify_obsolete_files(self):
        """Identify potentially obsolete or redundant files"""
        print("\n🔍 Analyzing for obsolete files...")
        
        obsolete_candidates = []
        
        # Check for duplicate functionality
        analyzed = self.migration_report['files_analyzed']
        
        # Group by category
        by_category = {}
        for file_info in analyzed:
            cat = file_info.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(file_info)
        
        # Look for similar files
        for category, files in by_category.items():
            if len(files) > 1:
                print(f"\n  Category '{category}' has {len(files)} files:")
                for f in files:
                    classes = ', '.join(f['classes'][:3]) if f['classes'] else 'None'
                    print(f"    - {Path(f['path']).name} ({f['line_count']} lines, classes: {classes})")
                
                print(f"    💡 Consider merging similar files in '{category}'")
        
        # Check for very small files
        small_files = [f for f in analyzed if f['line_count'] < 20 and not f['is_test']]
        if small_files:
            print(f"\n  Found {len(small_files)} small files (<20 lines) that might be merged:")
            for f in small_files[:5]:
                print(f"    - {Path(f['path']).name}")
        
        return obsolete_candidates
    
    def generate_report(self):
        """Generate migration report"""
        report_path = self.target_dir / 'migration_report.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.migration_report, f, indent=2)
        
        # Generate human-readable report
        readme_path = self.target_dir / 'MIGRATION_REPORT.md'
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Project Migration Report\n\n")
            f.write(f"**Date:** {self.migration_report['timestamp']}\n\n")
            f.write(f"**Source:** `{self.migration_report['source']}`\n\n")
            f.write(f"**Target:** `{self.migration_report['target']}`\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Files analyzed: {len(self.migration_report['files_analyzed'])}\n")
            f.write(f"- Files moved: {len(self.migration_report['files_moved'])}\n")
            f.write(f"- Imports updated: {len(self.migration_report['imports_updated'])}\n")
            f.write(f"- Errors: {len(self.migration_report['errors'])}\n\n")
            
            if self.migration_report['errors']:
                f.write("## Errors\n\n")
                for error in self.migration_report['errors']:
                    f.write(f"- **{error['file']}**: {error['error']}\n")
                f.write("\n")
            
            f.write("## Files Moved\n\n")
            for move in self.migration_report['files_moved'][:20]:
                from_path = Path(move['from']).name
                to_path = move['to']
                f.write(f"- `{from_path}` → `{to_path}`\n")
            
            if len(self.migration_report['files_moved']) > 20:
                f.write(f"\n... and {len(self.migration_report['files_moved']) - 20} more files\n")
        
        print(f"\n✓ Migration report saved: {readme_path}")
        print(f"✓ Detailed JSON report: {report_path}")
    
    def migrate(self):
        """Run full migration"""
        print("=" * 60)
        print("PROJECT MIGRATION TOOL")
        print("=" * 60)
        
        # Step 1: Backup
        self.backup_project()
        
        # Step 2: Analyze
        self.analyze_project()
        
        # Step 3: Identify obsolete
        self.identify_obsolete_files()
        
        # Step 4: Reorganize
        self.reorganize_files()
        
        # Step 5: Generate report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✓ MIGRATION COMPLETE!")
        print("=" * 60)
        print(f"\nBackup location: {self.backup_dir}")
        print(f"New structure: {self.target_dir}")
        print("\nNext steps:")
        print("1. Review migration_report.json")
        print("2. Test imports: python -m pytest tests/")
        print("3. Update any remaining imports manually")
        print("4. Delete obsolete files if confirmed")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate and reorganize RL project structure'
    )
    parser.add_argument('--source', type=str, required=True,
                        help='Source project directory')
    parser.add_argument('--target', type=str, default=None,
                        help='Target directory (default: source + _reorganized)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating backup')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    if args.target:
        target_dir = Path(args.target).resolve()
    else:
        target_dir = source_dir.parent / f"{source_dir.name}_reorganized"
    
    # Create migrator
    migrator = ProjectMigrator(
        source_dir=source_dir,
        target_dir=target_dir,
        backup=not args.no_backup
    )
    
    # Run migration
    migrator.migrate()


if __name__ == '__main__':
    main()
