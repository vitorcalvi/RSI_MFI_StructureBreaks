#!/usr/bin/env python3
"""
Script to patch f-strings in core/trade_engine.py and main.py to guard against None values
and avoid unsupported format string errors.
Auto-detects files in cwd:
  - core/trade_engine.py
  - main.py
"""
import re
from pathlib import Path


def patch_file(path: Path, pattern: str, replacement: str):
    text = path.read_text()
    new_text, count = re.subn(pattern, replacement, text)
    if count:
        path.write_text(new_text)
        print(f"Patched {count} occurrence(s) in {path}")
    else:
        print(f"No matches for pattern in {path}")


if __name__ == '__main__':
    base = Path.cwd()
    targets = {
        # Patch wallet balance print in trade_engine.py
        base / 'core' / 'trade_engine.py': {
            'pattern': r"print\(f\".*Wallet balance:.*\"\)",
            'replacement': (
                "bal = self.get_wallet_balance() or 0.0\n"
                "                print(f\"✅ Wallet balance: ${bal:,.2f}\")"
            )
        },
        # Patch display_startup_info print in main.py for any Balance formatting
        base / 'main.py': {
            'pattern': r"print\(f\"🚀.*Balance:.*\"\)",
            'replacement': (
                "    bal = wallet_balance or 0.0\n"
                "    price = current_price or 0.0\n"
                "    print(\n"
                "        f\"🚀 {symbol} Bot | {mode} | Balance: ${bal:,.0f}\"\n"
                "    )"
            )
        }
    }

    for path, info in targets.items():
        if not path.exists():
            print(f"File not found: {path}")
            continue
        patch_file(path, info['pattern'], info['replacement'])