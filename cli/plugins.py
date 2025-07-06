"""Plugins command group: manage plugins (list, enable, disable, install)"""
from tabulate import tabulate
from colorama import Fore, Style
from plugins.plugin_manager import PluginManager
from plugins.plugin_registry import PluginRegistry
import asyncio

# Singleton plugin manager for CLI
_plugin_manager = None
def get_plugin_manager():
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
        # Discover plugins at startup
        asyncio.run(_plugin_manager.discover_plugins())
    return _plugin_manager

def handle(args):
    manager = get_plugin_manager()
    registry = manager.registry
    if not args or args[0] == "list":
        # List all plugins
        plugins = registry.get_all_metadata()
        table = []
        for p in plugins:
            color = Fore.GREEN if p.enabled else Fore.RED
            table.append([
                color + p.name + Style.RESET_ALL,
                p.version,
                p.plugin_type.value,
                p.author,
                ", ".join(p.tags),
                p.description,
                "Enabled" if p.enabled else "Disabled"
            ])
        print(tabulate(table, headers=["Name", "Version", "Type", "Author", "Tags", "Description", "Status"]))
    elif args[0] == "enable":
        if len(args) < 2:
            print(Fore.RED + "Usage: plugins enable <plugin_name>" + Style.RESET_ALL)
            return
        plugin_name = args[1]
        meta = registry.get_plugin_metadata(plugin_name)
        if not meta:
            print(Fore.RED + f"Plugin '{plugin_name}' not found." + Style.RESET_ALL)
            return
        meta.enabled = True
        print(Fore.GREEN + f"Plugin '{plugin_name}' enabled." + Style.RESET_ALL)
    elif args[0] == "disable":
        if len(args) < 2:
            print(Fore.RED + "Usage: plugins disable <plugin_name>" + Style.RESET_ALL)
            return
        plugin_name = args[1]
        meta = registry.get_plugin_metadata(plugin_name)
        if not meta:
            print(Fore.RED + f"Plugin '{plugin_name}' not found." + Style.RESET_ALL)
            return
        meta.enabled = False
        print(Fore.YELLOW + f"Plugin '{plugin_name}' disabled." + Style.RESET_ALL)
    elif args[0] == "install":
        if len(args) < 2:
            print(Fore.RED + "Usage: plugins install <plugin_path>" + Style.RESET_ALL)
            return
        plugin_path = args[1]
        # Try to discover and register the plugin
        discovered = asyncio.run(manager.discover_plugins())
        if discovered:
            print(Fore.GREEN + f"Plugins discovered/installed from '{plugin_path}'." + Style.RESET_ALL)
        else:
            print(Fore.RED + f"No plugins found in '{plugin_path}'." + Style.RESET_ALL)
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: plugins <subcommand>
Subcommands:
  list                      List all plugins
  enable <plugin_name>      Enable a plugin
  disable <plugin_name>     Disable a plugin
  install <plugin_path>     Discover/install plugins from a path
""") 