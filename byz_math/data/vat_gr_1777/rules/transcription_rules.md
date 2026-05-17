# Transcription Rules — Vat.gr.1777
## Diplomatic Transcription Standard v1.0

---

## Guiding Principle

> **Transcribe what the scribe wrote — not what the text should say.**

A diplomatic transcription encodes the **visual signal of the manuscript surface** into Unicode text. It is NOT a critical edition, NOT a translation, and NOT a normalised text. Every deviation from these rules contaminates the ground truth for HTR training.

---

## Layer Definitions

| Layer | Definition | File suffix |
|-------|-----------|-------------|
| **Diplomatic** | Exact reproduction — abbreviations preserved, spelling preserved, no modernisation | `_dipl.txt` |
| **Expanded** | Abbreviations resolved, no other changes | `_exp.txt` |
| **Normalised** | Accentuation regularised, sigma/lunate unified | `_norm.txt` |
| **Translation** | Semantic rendition | `_trans.txt` |

**Always keep layers in separate files. Never mix them in a single file.**

---

## 1. Uncertainty Notation

| Situation | Diplomatic notation | Example |
|-----------|-------------------|---------|
| Single uncertain letter | `[x?]` | `[δ?]` |
| Two possible readings | `[x/y?]` | `[α/λ?]` |
| Completely illegible character | `[illeg.]` | `τ[illeg.]ν` |
| Multiple illegible characters | `[illeg. N]` | `[illeg. 3]` |
| Physical gap / lacuna | `[gap]` | |
| Gap of known extent | `[gap: N chars]` | `[gap: ~5 chars]` |
| Damaged but legible | `⟨x⟩` | `⟨δ⟩` (conjectural) |
| Cancelled text (expunged) | `{x}` | `{δε}` |
| Scribal addition (supralinear) | `^x^` | `^καί^` |
| Scribal addition (marginal) | `||x||` | `||nota bene||` |
| Text erased but readable (palimpsest) | `〚x〛` | |

---

## 2. Sigma / Lunate Sigma Rule

Vat.gr.1777 is a 15th-century manuscript. The scribe **may use lunate sigma (ϲ)** in some or all positions rather than medial σ / final ς.

**Rule:** Transcribe the exact form you see.
- If the letter looks like a C-shape → use ϲ (U+03F2)
- If the letter looks like a medial sigma (σ) → use σ
- If the letter looks like a final sigma (ς) → use ς
- If uncertain which sigma form → use `[σ?/ϲ?]`

**Never** automatically convert ϲ → σ/ς in the diplomatic layer.

---

## 3. Accents and Breathings

**Rule:** Preserve exactly what you see. Do not add or remove accents.

- If the manuscript shows an accent that differs from classical usage → preserve it
- If the accent appears smudged or absent → note: `[no accent?]` in the margin note, but do not add one to the diplomatic text
- Use precomposed forms from Greek Extended (U+1F00–U+1FFF) wherever possible

---

## 4. Abbreviations

### Diplomatic layer: PRESERVE
Write the abbreviation as it appears:
- `κ̅ς̅` (not κύριος)
- `ϰαί` ligature → encode best Unicode approximation + note in glyph inventory
- `ο̄ν` → encode with combining overline: ο + U+0305 + ν

### Encoding the overline
Letter + U+0305 (COMBINING OVERLINE):
```
κ̅ = κ + \u0305
ς̅ = ς + \u0305
```

### Expanded layer: RESOLVE
```
κ̅ς̅ → κύριος
θ̄ς̄ → θεός
```

---

## 5. Line Breaks and Structure

- End each line with a newline `\n`
- A folio/recto break: `--- [f. 1r] ---`
- Column breaks (if two-column): `=COL2=`
- Line number within folio: prefix `L01:`, `L02:`, etc.

**Example:**
```
--- [f. 1r] ---
L01: τοῖς ζητοῦσι τὴν ἀλήθεια[ν?]
L02: κατ᾽ ἐπιστήμην τεχν̄η
L03: [illeg. 2] ὡς ἔφαμεν ἐν τ̄ο̄ῑς
```

---

## 6. Ligatures

When a ligature cannot be represented in Unicode:
1. Encode the component letters separately
2. Add entry to `glyph_inventory.csv` with an ID
3. Reference: `{LIG:G045}` inline if needed for precision

Common ligatures that DO have Unicode representations:
- ου → often written as a single form; encode as ο + υ unless clearly a special glyph
- αι, ει → usually encodable
- καί → sometimes a special ligature; use ϰαί or note

---

## 7. Punctuation

Use the **Byzantine punctuation characters**:
- High dot (full stop) → U+0387 GREEK ANO TELEIA: `·`
- Question → U+037E GREEK QUESTION MARK: `;`
- Comma → U+002C
- Paragraph marker → preserve as `¶` or `⸏`

Do **not** insert modern punctuation not visible in the manuscript.

---

## 8. Greek Numerals

When you see a letter with a horizontal bar above it → it is a numeral.
- Encode as: letter + U+0305 (combining overline)
- Example: α with overline = `ᾱ` = 1; ιβ with overline = `ῑβ̄` = 12

For the right-keraia sign (single tick after number): use U+0374 `ʹ`
For the left-keraia (thousands): use U+0375 `͵`

---

## 9. What NOT to Do

| ❌ Forbidden | ✅ Correct |
|-------------|-----------|
| Correct spelling silently | Preserve the spelling you see |
| Add missing accents | Mark absence in notes |
| Expand abbreviations in diplomatic layer | Keep κ̅ς̅ as κ̅ς̅ |
| Convert ϲ → σ/ς automatically | Transcribe the form you see |
| Use modern Greek question mark (;) | Use U+037E |
| Add punctuation not in MS | Leave absent |
| Guess uncertain letters | Mark `[x?]` |
| Merge two uncertain letters | Mark each separately |

---

## 10. File Naming Convention

```
{folio}_{side}_{layer}.txt

Examples:
  0007_1r_dipl.txt      ← folio 1 recto, diplomatic
  0007_1r_exp.txt       ← folio 1 recto, expanded
  0007_1r_notes.md      ← paleographic notes for this folio

Line crops:
  0007_1r_L01.png       ← line 1 crop from folio 1r
  0007_1r_L02.png       ← line 2 crop
```

The leading 4-digit number matches the IIIF canvas number from `manifest_meta.json`.

---

## 11. Metadata Header (each diplomatic file)

Every `.txt` transcription file begins with this YAML header block:

```yaml
---
manuscript: Vat.gr.1777
folio: 1r
canvas: 0007
layer: diplomatic
transcriber: [your name]
date: YYYY-MM-DD
confidence: [high / medium / low]
notes: ""
---
```
