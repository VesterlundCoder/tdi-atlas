# Abbreviations Catalog — Byzantine Greek Manuscripts
## Reference for Vat.gr.1777 Diplomatic Transcription

---

## 1. Nomina Sacra

Standard contractions of sacred names, marked by a horizontal bar over the letters.

| Diplomatic | Expanded | Meaning | Frequency in Vat.gr.1777 |
|------------|----------|---------|--------------------------|
| θ̅ς̅ | θεός | God | Moderate (formulae) |
| κ̅ς̅ | κύριος | Lord | Moderate |
| χ̅ς̅ | Χριστός | Christ | Rare (philosophical MS) |
| ι̅ς̅ | Ἰησοῦς | Jesus | Rare |
| π̅ρ̅ | πατήρ | Father | Rare |
| υ̅ς̅ | υἱός | Son | Rare |
| π̅ν̅α̅ | πνεῦμα | Spirit | Rare |
| σ̅τ̅ρ̅ | σωτήρ | Saviour | Rare |
| ο̅υ̅ρ̅ | οὐρανός | Heaven | Rare |
| ἀ̅ν̅ο̅ς̅ | ἄνθρωπος | Human | Rare |

**Encoding:** Each overlined letter = base_letter + U+0305 (COMBINING OVERLINE)
```
κ̅ς̅ = κ\u0305ς\u0305
θ̅ς̅ = θ\u0305ς\u0305
```

---

## 2. Common Scribal Abbreviations (Philosophical / Logical Texts)

These appear frequently in Aristotelian commentary manuscripts of the 14th–15th centuries.

| Diplomatic | Expanded | Category | Notes |
|------------|----------|---------|-------|
| ο̅ν̅ | ον / ων | Ending | Neuter participle / genitive plural |
| τ̅ν̅ | τῶν | Grammar | Very common genitive |
| ε̅ | ἐν / ἐστί | Grammar | Context determines |
| ε̅σ̅τ̅ι̅ | ἐστί | Copula | Common in logical texts |
| κ̅αί | καί | Conjunction | Sometimes ligature |
| ο̅υ̅ν̅ | οὖν | Conjunction | Common in argument |
| ε̅ι̅ | εἰ | Conditional | In syllogistic |
| ο̅τ̅ι̅ | ὅτι | Conjunction | |
| ε̅ν̅ | ἐν | Preposition | |
| α̅λ̅λ̅α̅ | ἀλλά | Adversative | |
| μ̅ε̅ν̅ | μέν | Particle | |
| δ̅ε̅ | δέ | Particle | |
| π̅ρ̅ο̅ς̅ | πρός | Preposition | |
| π̅ε̅ρ̅ι̅ | περί | Preposition | |
| κ̅α̅τ̅α̅ | κατά | Preposition | |

---

## 3. Aristotelian / Logical Technical Abbreviations

Specific to philosophical commentaries (Ammonius, Argyros, Gennadius).

| Diplomatic | Expanded | Meaning |
|------------|----------|---------|
| ἀρ̅ | ἄρα | Therefore (logical conclusion) |
| συλ̅ | συλλογισμός | Syllogism |
| πρ̅ | πρότασις | Proposition / Premise |
| συμπ̅ | συμπέρασμα | Conclusion |
| κατ̅ | κατηγορία | Category / Predicate |
| ὑπ̅ | ὑποκείμενον | Subject |
| κτλ̅ | καὶ τὰ λοιπά | Et cetera |
| ὁρ̅ | ὁρισμός | Definition |
| γ̅ν̅ | γνώμη / γνῶσις | Judgement / Knowledge |

---

## 4. Suspension Abbreviations

The scribe writes the start of a word and omits the rest, marked by overline or raised dot.

| Form | Probable expansion | Rule |
|------|-------------------|------|
| First 2–3 letters + ̅ | Any common word | Must check context |
| τ̅ | τόν / τῶν / τό | Determine by syntax |
| ἐ̅ | ἐν / ἐκ / ἐπί / ἐστί | Determine by context |
| π̅ | πᾶς / πρός / περί / πατήρ | Determine by context |

**RULE:** In the diplomatic layer, always mark suspended letters as-is with the overline. In the expanded layer, give the most likely expansion with a confidence note.

---

## 5. Ligatures — Common Byzantine Forms

These character combinations are often written as a single joined glyph.

| Ligature (description) | Unicode encoding | Notes |
|-----------------------|-----------------|-------|
| ου (joined) | ο + υ | Write as two characters; note in glyph inventory if unusual form |
| αι (joined) | α + ι | Two characters |
| ει (joined) | ε + ι | Two characters |
| καί (special ligature) | κ + α + ί | Sometimes a special glyph; add to glyph inventory |
| γρ (joined) | γ + ρ | Common in 15th c. |
| στ (joined) | σ + τ | Common |
| τε (joined) | τ + ε | Common |
| Final -ων (joined loop) | ω + ν | Common in scholarly hands |
| Initial ει- (arch) | ε + ι | Initial position form |

**If a ligature cannot be encoded:** write component letters + add `{LIG:G[ID]}` reference pointing to your glyph inventory entry.

---

## 6. Superscript / Raised Letters

In Byzantine manuscripts, smaller raised letters sometimes abbreviate suffixes or endings.

| Diplomatic form | Meaning |
|----------------|---------|
| word + raised ν | Final -ν (n-ephelkustikon or -ν ending) |
| word + raised ς | Final -ς |
| word + raised ι | Subscript iota (historically raised in some MSS) |

Encode raised letters with: `^ν^` in diplomatic layer.

---

## 7. Punctuation Marks That Look Like Abbreviation Marks

| Symbol | Do NOT confuse with | Correct interpretation |
|--------|--------------------|-----------------------|
| ´ (right keraia, U+0374) | Accent | Numeral sign |
| ͵ (left keraia, U+0375) | Comma | Thousands marker |
| · (high dot) | Abbreviation mark | Punctuation stop |
| ¯ (horizontal line above) | Macron | Overline = abbreviation OR numeral |

---

## 8. Catalog Maintenance

When you encounter a new abbreviation not in this catalog:
1. Note it as `[ABB:???]` in the diplomatic text
2. Add a row to this file with all known info
3. Mark status: `CONFIRMED` / `HYPOTHETICAL` / `CONTEXT-ONLY`

| Date added | Folio | Diplomatic form | Proposed expansion | Status |
|------------|-------|----------------|-------------------|--------|
| — | — | — | — | — |
