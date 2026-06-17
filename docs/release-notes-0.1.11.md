# RetroSynFormer 0.1.11 Release Notes

*Branch: `feature-nonuniform-dropout` — June 2026*

---

## Summary

Fixed misaligned verbose training output: column values now print in the same
order as their headers, columns are fixed-width so they stay aligned regardless
of terminal tab settings, and the header row reprints automatically every 10
epochs.

---

## Changes

### 1. Verbose training column alignment (`src/retrosynformer/trainer.py`)

**Bug fix — `study` column order:**  
`study_name` was being prepended to each row (appearing before `epoch`) while
the header printed it as a suffix (after `note`). Both now use a suffix
position, so `study` is always the rightmost column in both header and row.

**Fixed-width columns:**  
Replaced `"\t".join(...)` with `"  ".join(...)` using explicit format-string
widths:

| Column | Width | Alignment |
|--------|-------|-----------|
| `trial` | 5 | right |
| `epoch` | 5 | right |
| `t_loss`, `t_acc`, `t_racc`, `v_loss`, `v_acc`, `v_racc` | 7 | right |
| `s/ep` | 6 | right |
| `note` | 4 | left |
| `study` | variable | left |

Tab stops vary by terminal width and font; fixed-width fields do not.

**Periodic header reprint:**  
The header line reprints before the row at every 10th epoch relative to
`start_epoch` (i.e. at `start_epoch + 10`, `start_epoch + 20`, …). This
keeps column labels visible during long runs without scrolling back.

The header print is factored into a `_print_header()` closure (defined once
before the epoch loop, capturing `_hdr_prefix` and `_hdr_suffix`) so the
format is defined in exactly one place.
