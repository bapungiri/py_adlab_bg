# groupdata_slots_generator.py

import pathlib
import re


def generate_slots(results_dir: pathlib.Path, out_file: pathlib.Path):
    """Regenerate groupdata_slots.py based on .npy files in results_dir."""

    def make_identifier(name: str) -> str:
        ident = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        if ident and ident[0].isdigit():
            ident = "f_" + ident
        return ident

    slots = ["path"]

    for f in results_dir.glob("*.npy"):
        slots.append(make_identifier(f.stem))

    slots_sorted = sorted(set(slots), key=str.lower)

    out_file.write_text(
        "# Auto-generated — Do not edit manually\n"
        "SLOTS = (\n" + "".join(f'    "{s}",\n' for s in slots_sorted) + ")\n",
        encoding="utf-8",
    )

    print(f"[GroupData] Regenerated {out_file} ({len(slots_sorted)} entries).")
