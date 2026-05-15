# Diophantus — Notation Reference Card

**Manuscript:** Paris, BnF, Grec 2485  
**Siglum:** A (Tannery)  
**Date:** 1560–70, Italy (Rome/Ferrara)  
**Folios:** 214  
**Contains:** Arithmetica I–VI + On Polygonal Numbers + Planudes scholia  

---

## Gallica Links (open in browser)

| What | URL |
|------|-----|
| Full manuscript viewer | https://gallica.bnf.fr/ark:/12148/btv1b10722263g |
| IIIF manifest (machine) | https://gallica.bnf.fr/iiif/ark:/12148/btv1b10722263g/manifest.json |
| BnF catalog entry | http://archivesetmanuscrits.bnf.fr/ark:/12148/cc102994q |
| Heath 1910 (archive.org) | https://archive.org/details/diophantusofalex00heatiala |
| Tannery 1893 Vol.1 (archive.org) | https://archive.org/details/diophantialexan01plangoog |

**Direct image URL pattern:**  
`https://gallica.bnf.fr/iiif/ark:/12148/btv1b10722263g/f{N}/full/500,/0/native.jpg`  
Replace `{N}` with folio number (1–214).

---

## The Notation System

### Unknown & Powers of x

| Symbol in MS | Name (Greek) | Meaning | Unicode approx. |
|---|---|---|---|
| ϛ | ἀριθμός (*arithmos*) | x (the unknown) | U+03DB ϛ |
| ΔΥ | Δύναμις (*Dynamis*) | x² | ΔΥ (2 chars) |
| ΚΥ | Κύβος (*Kybos*) | x³ | ΚΥ (2 chars) |
| ΔΔ | Δυναμοδύναμις | x⁴ | ΔΔ |
| ΔΚ | Δυναμόκυβος | x⁵ | ΔΚ |
| ΚΚ | Κυβόκυβος | x⁶ | ΚΚ |

### Inverse Powers (reciprocals)

| Symbol | Name | Meaning |
|---|---|---|
| ϛΑ | ἀριθμοστόν | 1/x |
| ΔΑ | Δυναμοστόν | 1/x² |
| ΚΑ | Κυβοστόν | 1/x³ |
| ΔΔΑ | Δυναμοδυναμοστόν | 1/x⁴ |
| ΔΚΑ | Δυναμοκυβοστόν | 1/x⁵ |
| ΚΚΑ | Κυβοκυβοστόν | 1/x⁶ |

### Constants & Operations

| Symbol | Meaning | Note |
|---|---|---|
| Μ (mu) | Μονάδες = units/constant | e.g. Μ γ̄ = 3 |
| *(juxtaposition)* | addition | no sign; positive terms first |
| ↗ *(inverted truncated ψ)* | subtraction sign | **no Unicode codepoint** — marks all subtracted terms as a group |
| ἴσ | equals | abbreviation of ἴσος |

> **Critical gap:** The minus sign has NO Unicode codepoint. This is the primary encoding challenge.  
> **Usage:** All terms after ↗ are subtracted as a group.  
> Example: `ΚΥ α ΔΥ γ [↗] ϛ ε Μ β` = x³ + 3x² − 5x − 2

---

## ϛ Ambiguity — Key Point for ML

The symbol **ϛ (stigma, U+03DB)** has **two distinct meanings** in Diophantus:

1. **Numeral 6** (in Milesian system: ϛ̄ = 6)
2. **Algebraic unknown** (x)

Context determines reading. In equations, ϛ is the unknown; in coefficient lists, it is the numeral 6.  
**This ambiguity is unresolved in any existing digital encoding** — a direct research target.

---

## Milesian Numerals Quick Reference

| Units | | Tens | | Hundreds | |
|---|---|---|---|---|---|
| ᾱ=1 | δ̄=4 | ζ̄=7 | ῑ=10 | μ̄=40 | ρ̄=100 |
| β̄=2 | ε̄=5 | η̄=8 | κ̄=20 | ν̄=50 | σ̄=200 |
| γ̄=3 | ϛ̄=6 | θ̄=9 | λ̄=30 | ξ̄=60 | τ̄=300 |

Overline (U+0305 COMBINING OVERLINE) marks numerals. Μ alone = 10,000 (myriad).

---

## Example Expressions

```
ΔΥ ᾱ ϛ γ̄ Μ β̄          →  x² + 3x + 2
ΔΥ δ̄ ἴσ ΔΥ β̄ ϛ ϛ̄      →  4x² = 2x² + 6x   (note: ϛ̄ here is numeral 6)
ΚΥ ᾱ [↗] Μ η̄           →  x³ − 8
```

---

## Planudes Scholia

Maximus Planudes (1260–1310) added marginal/interlinear scholia in **full-page width** (Diophantus text is in two columns). Identified by the sigil **σχ** at the start of each scholion.

These are the **earliest Byzantine mathematical commentary on Diophantus** and an important research target.

---

## TEI/XML Encoding Proposal

```xml
<!-- Namespace: xmlns:dioph="http://tdi-atlas.research/diophantus/notation/1.0" -->

<!-- x² + 3x + 2 -->
<seg type="dioph:expr">
  <seg type="dioph:power" n="2">ΔΥ</seg> <num>ᾱ</num>
  <seg type="dioph:power" n="1">ϛ</seg> <num>γ̄</num>
  <seg type="dioph:constant">Μ</seg> <num>β̄</num>
</seg>

<!-- Subtraction sign (no Unicode): -->
<pc type="dioph:minus">
  <desc>Diophantus minus sign — inverted truncated psi, no Unicode codepoint</desc>
</pc>

<!-- Disambiguation of ϛ: -->
<seg type="dioph:unknown" n="1">ϛ</seg>   <!-- x -->
<num value="6">ϛ</num>                    <!-- numeral 6 -->
```

**Status:** No TEI standard for Diophantus notation exists. The scheme above is a research proposal.

---

## Files in This Directory

| File | Contents |
|------|----------|
| `manuscript_record.json` | Full codicological metadata + IIIF links |
| `notation_encoding.json` | Complete machine-readable notation table (JSON) |
| `DIOPHANTUS_REFERENCE.md` | This file — human-readable reference card |
| `fetch_manuscript.py` | Script to download IIIF images from Gallica |
| `images/` | Downloaded folio images (created by fetch script) |

---

## Key Scholars

| Scholar | Contribution |
|---------|-------------|
| Paul Tannery | Critical edition 1893–95; established manuscript tradition |
| T.L. Heath | English study + translation 1910; definitive notation analysis |
| André Allard | Corrected Tannery's scribal-hand misattributions; Planudes scholia study |
| Fabio Acerbi | Byzantine recensions survey (HAL hal-03596963) |
| Isabella Grigoriadis | Recent work on Planudes mathematics |
