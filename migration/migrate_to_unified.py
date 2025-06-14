"""
Migration Script for Unified System
Helps transition from the old system to the new unified system.
"""
import os
import sys
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config import get_unified_config_manager
from auth.unified_auth import get_unified_auth_manager, UserRole
from data.unified_data_layer import get_unified_data_layer, EntityType


class UnifiedMigration:
    """Migration manager for transitioning to unified system."""
    
    def __init__(self):
        """Initialize the migration manager."""
        self.logger = logging.getLogger("migration")
        self.backup_dir = Path("migration/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration status
        self.migration_log: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        
        self.logger.info("Migration manager initialized")
    
    def create_backup(self) -> bool:
        """Create backup of existing configuration and data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            self.logger.info(f"Creating backup at {backup_path}")
            
            # Backup existing config files
            config_files = [
                "config/web_config.py",
                "config/langchain_config.py", 
                "cli/config_manager.py",
                "config/settings.py"
            ]
            
            for config_file in config_files:
                if Path(config_file).exists():
                    shutil.copy2(config_file, backup_path / Path(config_file).name)
                    self.logger.info(f"Backed up {config_file}")
            
            # Backup existing data directories
            data_dirs = ["output", "logs", "data"]
            for data_dir in data_dirs:
                if Path(data_dir).exists():
                    shutil.copytree(data_dir, backup_path / data_dir, dirs_exist_ok=True)
                    self.logger.info(f"Backed up {data_dir}")
            
            # Create backup manifest
            manifest = {
                "backup_created": datetime.now().isoformat(),
                "files_backed_up": config_files,
                "directories_backed_up": data_dirs,
                "migration_version": "2.0.0"
            }
            
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.migration_log.append({
                "step": "backup_created",
                "timestamp": datetime.now().isoformat(),
                "backup_path": str(backup_path),
                "status": "success"
            })
            
            self.logger.info("✅ Backup created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            self.errors.append(f"Backup failed: {e}")
            return False
    
    def migrate_configuration(self) -> bool:
        """Migrate existing configuration to unified config."""
        try:
            self.logger.info("🔄 Migrating configuration...")
            
            config_manager = get_unified_config_manager()
            unified_config = config_manager.get_config()
            
            # Migrate CLI configuration
            self._migrate_cli_config(unified_config)
            
            # Migrate web configuration
            self._migrate_web_config(unified_config)
            
            # Migrate LangChain configuration
            self._migrate_langchain_config(unified_config)
            
            # Save unified configuration
            config_manager.save_configuration()
            
            self.migration_log.append({
                "step": "configuration_migrated",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
            self.logger.info("✅ Configuration migrated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate configuration: {e}")
            self.errors.append(f"Configuration migration failed: {e}")
            return False
    
    def _migrate_cli_config(self, unified_config) -> None:
        """Migrate CLI configuration."""
        try:
            # Check for existing CLI config
            cli_config_file = Path("config/cli_config.yaml")
            if cli_config_file.exists():
                with open(cli_config_file, 'r') as f:
                    cli_data = yaml.safe_load(f)
                
                # Merge CLI data into unified config
                if cli_data:
                    unified_config.cli.update(cli_data)
                    self.logger.info("CLI configuration migrated")
            
        except Exception as e:
            self.logger.warning(f"CLI config migration warning: {e}")
    
    def _migrate_web_config(self, unified_config) -> None:
        """Migrate web configuration."""
        try:
            # Web config is already part of unified config structure
            # Just ensure defaults are set
            if not unified_config.web.title:
                unified_config.web.title = "Unified Web Scraper System"
            
            self.logger.info("Web configuration migrated")
            
        except Exception as e:
            self.logger.warning(f"Web config migration warning: {e}")
    
    def _migrate_langchain_config(self, unified_config) -> None:
        """Migrate LangChain configuration."""
        try:
            # LangChain config is already part of unified config structure
            self.logger.info("LangChain configuration migrated")
            
        except Exception as e:
            self.logger.warning(f"LangChain config migration warning: {e}")
    
    def migrate_data(self) -> bool:
        """Migrate existing data to unified data layer."""
        try:
            self.logger.info("🗄️ Migrating data...")
            
            data_layer = get_unified_data_layer()
            
            # Migrate existing job data
            self._migrate_job_data(data_layer)
            
            # Migrate existing log data
            self._migrate_log_data(data_layer)
            
            # Migrate existing result data
            self._migrate_result_data(data_layer)
            
            self.migration_log.append({
                "step": "data_migrated",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
            self.logger.info("✅ Data migrated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate data: {e}")
            self.errors.append(f"Data migration failed: {e}")
            return False
    
    def _migrate_job_data(self, data_layer) -> None:
        """Migrate existing job data."""
        try:
            # Look for existing job files
            output_dir = Path("output")
            if output_dir.exists():
                for job_file in output_dir.glob("*.json"):
                    try:
                        with open(job_file, 'r') as f:
                            job_data = json.load(f)
                        
                        # Create job entity
                        data_layer.create_entity(
                            entity_type=EntityType.JOB,
                            data={
                                "name": job_file.stem,
                                "job_type": "web_scraping",
                                "status": "completed",
                                "config": job_data,
                                "progress": 100,
                                "total_tasks": 1,
                                "completed_tasks": 1
                            },
                            metadata={
                                "migrated_from": str(job_file),
                                "migration_timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        self.logger.info(f"Migrated job data from {job_file}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate {job_file}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Job data migration warning: {e}")
    
    def _migrate_log_data(self, data_layer) -> None:
        """Migrate existing log data."""
        try:
            # Look for existing log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        # Create log entity for the file
                        data_layer.create_entity(
                            entity_type=EntityType.LOG,
                            data={
                                "type": "legacy_log",
                                "log_file": str(log_file),
                                "size_bytes": log_file.stat().st_size,
                                "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                            },
                            metadata={
                                "migrated_from": str(log_file),
                                "migration_timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        self.logger.info(f"Migrated log reference for {log_file}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate log {log_file}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Log data migration warning: {e}")
    
    def _migrate_result_data(self, data_layer) -> None:
        """Migrate existing result data."""
        try:
            # Look for existing result files
            results_patterns = ["output/*.csv", "output/*.xlsx", "output/*.sqlite"]
            
            for pattern in results_patterns:
                for result_file in Path(".").glob(pattern):
                    try:
                        # Create result entity
                        data_layer.create_entity(
                            entity_type=EntityType.RESULT,
                            data={
                                "type": "legacy_result",
                                "file_path": str(result_file),
                                "file_type": result_file.suffix[1:],  # Remove the dot
                                "size_bytes": result_file.stat().st_size,
                                "modified": datetime.fromtimestamp(result_file.stat().st_mtime).isoformat()
                            },
                            metadata={
                                "migrated_from": str(result_file),
                                "migration_timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        self.logger.info(f"Migrated result data from {result_file}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate result {result_file}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Result data migration warning: {e}")
    
    def create_default_users(self) -> bool:
        """Create default users for the unified system."""
        try:
            self.logger.info("👤 Creating default users...")
            
            auth_manager = get_unified_auth_manager()
            
            # Admin user should already exist from auth manager initialization
            admin_user = auth_manager.get_user_by_username("admin")
            if admin_user:
                self.logger.info("Admin user already exists")
            
            # Create a demo user
            try:
                demo_user = auth_manager.create_user(
                    username="demo",
                    password="demo123",
                    email="demo@webscraper.local",
                    role=UserRole.USER
                )
                self.logger.info("Demo user created")
            except ValueError:
                self.logger.info("Demo user already exists")
            
            self.migration_log.append({
                "step": "users_created",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
            self.logger.info("✅ Default users created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create default users: {e}")
            self.errors.append(f"User creation failed: {e}")
            return False
    
    def run_full_migration(self) -> bool:
        """Run the complete migration process."""
        try:
            self.logger.info("🚀 Starting full migration to unified system...")
            
            # Step 1: Create backup
            if not self.create_backup():
                return False
            
            # Step 2: Migrate configuration
            if not self.migrate_configuration():
                return False
            
            # Step 3: Migrate data
            if not self.migrate_data():
                return False
            
            # Step 4: Create default users
            if not self.create_default_users():
                return False
            
            # Create migration report
            self._create_migration_report()
            
            self.logger.info("🎉 Migration completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self.errors.append(f"Migration failed: {e}")
            return False
    
    def _create_migration_report(self) -> None:
        """Create a migration report."""
        try:
            report = {
                "migration_completed": datetime.now().isoformat(),
                "migration_version": "2.0.0",
                "steps_completed": self.migration_log,
                "errors": self.errors,
                "status": "success" if not self.errors else "completed_with_warnings"
            }
            
            report_file = self.backup_dir / "migration_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Migration report created: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create migration report: {e}")


def main():
    """Main migration function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    migration = UnifiedMigration()
    
    print("🔄 Starting migration to Unified Web Scraper System...")
    print("=" * 60)
    
    success = migration.run_full_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Review the migration report in migration/backups/")
        print("2. Test the unified system: python unified_system.py")
        print("3. Start the web interface: python start_web_interface.py")
        print("4. Use the unified CLI: python main.py")
    else:
        print("\n❌ Migration failed!")
        print("Check the logs for details and resolve any issues.")
        print("Your original files are backed up in migration/backups/")


if __name__ == "__main__":
    main()
