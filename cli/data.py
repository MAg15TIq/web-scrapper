"""Data command group: data export/import/clean/validate"""
from tabulate import tabulate
from colorama import Fore, Style
from data.unified_data_layer import get_unified_data_layer, EntityType
from utils.text_processing import AdvancedTextProcessor
from utils.data_validation import DataValidator
import json
import os

layer = get_unified_data_layer()
text_processor = AdvancedTextProcessor()
data_validator = DataValidator()

def handle(args):
    if not args or args[0] == "export":
        # Export all data to JSON
        all_entities = []
        for etype in EntityType:
            all_entities.extend(layer.list_entities(etype))
        out_file = args[1] if len(args) > 1 else "export.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in all_entities], f, indent=2)
        print(Fore.GREEN + f"Exported {len(all_entities)} entities to {out_file}." + Style.RESET_ALL)
    elif args[0] == "import":
        if len(args) < 2:
            print(Fore.RED + "Usage: data import <file>" + Style.RESET_ALL)
            return
        in_file = args[1]
        if not os.path.exists(in_file):
            print(Fore.RED + f"File '{in_file}' not found." + Style.RESET_ALL)
            return
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for entity in data:
            try:
                etype = EntityType(entity.get("entity_type", "DATA"))
                layer.create_entity(etype, entity.get("data", {}), entity_id=entity.get("id"), metadata=entity.get("metadata", {}))
                count += 1
            except Exception:
                continue
        print(Fore.GREEN + f"Imported {count} entities from {in_file}." + Style.RESET_ALL)
    elif args[0] == "clean":
        if len(args) < 2:
            print(Fore.RED + "Usage: data clean <text>" + Style.RESET_ALL)
            return
        text = args[1]
        cleaned = text_processor.clean_text(text)
        print(Fore.GREEN + "Cleaned text:" + Style.RESET_ALL)
        print(cleaned)
    elif args[0] == "validate":
        if len(args) < 3:
            print(Fore.RED + "Usage: data validate <value> <type>" + Style.RESET_ALL)
            return
        value = args[1]
        dtype = args[2]
        result = data_validator.validate_data_type(value, dtype)
        print(Fore.GREEN + "Validation result:" + Style.RESET_ALL)
        print(json.dumps(result, indent=2))
    else:
        print(Fore.RED + f"Unknown subcommand: {' '.join(args)}" + Style.RESET_ALL)
        help()

def help():
    print("""
Usage: data <subcommand>
Subcommands:
  export [file]              Export all data to a JSON file
  import <file>              Import data from a JSON file
  clean <text>               Clean a text string
  validate <value> <type>    Validate a value (type: string, int, float, email, url, etc.)
""") 