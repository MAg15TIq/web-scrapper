"""Config command group: view, edit, backup, restore configuration"""
from tabulate import tabulate
from colorama import Fore, Style
from config.unified_config import get_unified_config_manager
import os
import shutil

manager = get_unified_config_manager()


def handle(args):
    if not args or args[0] == "view":
        # View current config (summary)
        config = manager.get_config()
        table = [[k, v] for k, v in config.model_dump().items()]
        print(tabulate(table, headers=["Key", "Value"]))
    elif args[0] == "edit":
        if len(args) < 3:
            print(Fore.RED + "Usage: config edit <section.key> <value>" + Style.RESET_ALL)
            return
        section_key = args[1]
        value = args[2]
        keys = section_key.split('.')
        config = manager.get_config()
        d = config
        for k in keys[:-1]:
            d = getattr(d, k)
        # Warn if editing a complex type
        if hasattr(getattr(d, keys[-1]), '__dict__'):
            print(Fore.YELLOW + f"Warning: Editing complex/nested objects via CLI may not be fully supported." + Style.RESET_ALL)
        try:
            setattr(d, keys[-1], type(getattr(d, keys[-1]))(value))
            manager.save_configuration()
            print(Fore.GREEN + f"Config '{section_key}' updated to '{value}'." + Style.RESET_ALL)
            # Validate after edit
            errors = manager.validate_configuration()
            if errors:
                print(Fore.RED + "Validation errors after edit:" + Style.RESET_ALL)
                for err in errors:
                    print(Fore.RED + f"  - {err}" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Configuration is valid." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Failed to update config: {e}" + Style.RESET_ALL)
    elif args[0] == "backup":
        # Create a backup
        manager.save_configuration(backup=True)
        print(Fore.GREEN + "Configuration backup created." + Style.RESET_ALL)
    elif args[0] == "restore":
        if len(args) < 2:
            print(Fore.RED + "Usage: config restore <backup_file>" + Style.RESET_ALL)
            return
        backup_file = args[1]
        config_file = manager.config_file
        if not os.path.exists(backup_file):
            print(Fore.RED + f"Backup file '{backup_file}' not found." + Style.RESET_ALL)
            return
        shutil.copy2(backup_file, config_file)
        manager.reload_configuration()
        print(Fore.GREEN + f"Configuration restored from '{backup_file}'." + Style.RESET_ALL)
        # Validate after restore
        errors = manager.validate_configuration()
        if errors:
            print(Fore.RED + "Validation errors after restore:" + Style.RESET_ALL)
            for err in errors:
                print(Fore.RED + f"  - {err}" + Style.RESET_ALL)
        else:
            print(Fore.GREEN + "Configuration is valid." + Style.RESET_ALL)
    elif args[0] == "list-backups":
        # List all backup files
        backup_dir = manager.backup_dir
        if not backup_dir.exists():
            print(Fore.YELLOW + "No backup directory found." + Style.RESET_ALL)
            return
        backups = sorted(backup_dir.glob("*.yaml"))
        if not backups:
            print(Fore.YELLOW + "No backup files found." + Style.RESET_ALL)
            return
        print(Fore.GREEN + "Available backup files:" + Style.RESET_ALL)
        for b in backups:
            print(f"  {b.name}")
    elif args[0] == "delete-backup":
        if len(args) < 2:
            print(Fore.RED + "Usage: config delete-backup <backup_file>" + Style.RESET_ALL)
            return
        backup_file = manager.backup_dir / args[1]
        if not backup_file.exists():
            print(Fore.RED + f"Backup file '{backup_file}' not found." + Style.RESET_ALL)
            return
        try:
            backup_file.unlink()
            print(Fore.GREEN + f"Backup file '{backup_file.name}' deleted." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Failed to delete backup: {e}" + Style.RESET_ALL)
    elif args[0] == "add-watcher":
        print(Fore.YELLOW + "Watcher management is an advanced/internal feature. Use the Python API for full support." + Style.RESET_ALL)
    elif args[0] == "remove-watcher":
        print(Fore.YELLOW + "Watcher management is an advanced/internal feature. Use the Python API for full support." + Style.RESET_ALL)
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: config <subcommand>
Subcommands:
  view                        View current configuration
  edit <section.key> <value>  Edit a configuration value
  backup                      Create a configuration backup
  restore <backup_file>       Restore configuration from backup
  list-backups                List all available backup files
  delete-backup <backup_file> Delete a backup file
  add-watcher                 (Advanced) Add a config watcher (Python API only)
  remove-watcher              (Advanced) Remove a config watcher (Python API only)
""") 