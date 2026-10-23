# Animal records

Documentation only; analysis choices (which animals, quality, exclude windows)
live in `mab_subjects.py`.

Dates are `YYYY-MM-DD`, log times `YYYY-MM-DD HH:MM`. Don't store anything that
can be computed (age, phase length).

| File | One row per | Notes |
|---|---|---|
| `animals.csv` | animal | Fixed facts plus starting task setup. `rig_ip` is the first box; moves, valve changes and lesions go in the log. `end_reason`: sacrificed / perfused / died_in_surgery. |
| `surgeries/<animal>_<date>.md` | surgery | Copy `_template.md`. Plan images in `surgeries/images/`. |
| `log.csv` | event | `animal_id` may be `ALL`. `category`: health / task / hardware / software / housing / surgery. |

Edit in VS Code, not Excel: Excel rewrites dates when it saves a CSV.

Migrated 2026-10-22 from `Animal_info.xlsx`, `Lesioned_animals_info.xlsx` and the
per-animal `BG*.xlsx` notes in OneDrive `ADLab/experiments`.
