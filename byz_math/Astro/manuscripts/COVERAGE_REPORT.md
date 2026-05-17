Byzantine Astronomical Tables — Model Coverage Report
=====================================================
Generated: see scan_index.json

Models trained on: Vat.gr.1291, Klima V (Byzantium 41°N), Handy Tables Oblique Ascensions
Training RMSE (uncertain cells): flat: anaph=0.80°, hour=0.59h, chron=20.9min
                                  topo: anaph=3.39°, hour=0.35h, chron=11.8min
Ensemble (best-of-both):          flat→anaph 0.80°, topo→hour 0.35h, topo→chron 11.8min

Legend: █ TRANSCRIBABLE  ▒ UNCERTAIN  ░ OFF TABLE / INCOMPATIBLE

Coverage Categories
-------------------
TRANSCRIBABLE  = model RMSE small enough to identify correct value with high confidence
                 (anaph < 0.5°, hour < 0.3h, chron < 5min)
UNCERTAIN      = model prediction is in the right ballpark; manual check recommended
                 (anaph 0.5–2°, hour 0.3–1h, chron 5–15min)
OFF TABLE      = model predictions are unreliable for this column/manuscript

Manuscripts where models apply (directly or partially)
------------------------------------------------------

🟢  Vat.gr.1291  [HIGH confidence]
   Table types: handy_tables_oblique_ascensions, handy_tables_right_ascensions, handy_tables_declinations, handy_tables_solar_tables, almagest_star_catalogue
   FLAT  : ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   33.3% T   66.7% U    0.0% O
   TOPO  : ████████████████████▒▒▒▒▒▒▒▒▒▒   66.7% T   33.3% U    0.0% O
   ENSEMBL: ██████████████████████████████  100.0% T    0.0% U    0.0% O

🟢  Vat.gr.1594  [HIGH confidence]
   Table types: handy_tables_oblique_ascensions, almagest_tables
   FLAT  : ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   33.3% T   66.7% U    0.0% O
   TOPO  : ████████████████████▒▒▒▒▒▒▒▒▒▒   66.7% T   33.3% U    0.0% O
   ENSEMBL: ██████████████████████████████  100.0% T    0.0% U    0.0% O

🟢  Vat.gr.208  [HIGH confidence]
   Table types: handy_tables_oblique_ascensions
   FLAT  : ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   33.3% T   66.7% U    0.0% O
   TOPO  : ████████████████████▒▒▒▒▒▒▒▒▒▒   66.7% T   33.3% U    0.0% O
   ENSEMBL: ██████████████████████████████  100.0% T    0.0% U    0.0% O

🟢  Vat.gr.184  [HIGH confidence]
   Table types: handy_tables_oblique_ascensions, theon_scholia
   FLAT  : ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   33.3% T   66.7% U    0.0% O
   TOPO  : ████████████████████▒▒▒▒▒▒▒▒▒▒   66.7% T   33.3% U    0.0% O
   ENSEMBL: ██████████████████████████████  100.0% T    0.0% U    0.0% O

🟡  Vat.gr.211  [MEDIUM confidence]
   Table types: handy_tables_oblique_ascensions, persian_byzantine_tables
   FLAT  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   TOPO  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   ENSEMBL: ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    0.0% T  100.0% U    0.0% O

🟡  Vat.gr.198  [MEDIUM confidence]
   Table types: astronomical_tables_mixed, theon_commentary_excerpts
   FLAT  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   TOPO  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   ENSEMBL: ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    0.0% T  100.0% U    0.0% O

🟡  Vat.pal.gr.278  [MEDIUM confidence]
   Table types: paradosis_argyros, handy_tables_derived
   FLAT  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   TOPO  : ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░    0.0% T   66.7% U   33.3% O
   ENSEMBL: ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    0.0% T  100.0% U    0.0% O

Manuscripts where models do NOT apply
-------------------------------------

🔴  Vat.gr.792  [LOW]
   Table types: persian_byzantine_tribiblos, mean_motion_tables, equation_tables, planetary_tables
   Reason: Persian-Byzantine astronomical system (Shah Tables / Arjabhar). Different epoch, different mathematical basis. Models predict wrong absolute values. Can only verify tabular structure (monotonicity of cumulative columns, etc.).
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O
   → Requires new training data or general HTR (Kraken/Transkribus)

🔴  Vat.gr.1058  [LOW]
   Table types: persian_syntaxis_chrysokokkes, mean_motion_tables
   Reason: Persian Syntaxis (Zij-based): different table structure, different epoch (1283 Yazdigird era), different mean motions. Models predict wrong values.
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O
   → Requires new training data or general HTR (Kraken/Transkribus)

🔴  Vat.gr.1059  [LOW]
   Table types: persian_byzantine_tribiblos, mean_motion_tables, equation_tables
   Reason: Same Persian-Byzantine content as 792. Different scribal hand — valuable for cross-hand study of Persian tables.
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O
   → Requires new training data or general HTR (Kraken/Transkribus)

⚫  Vat.gr.1056  [NONE]
   Table types: arabo_byzantine_tables, arabic_star_lists_greek, astrological_tables
   Reason: Arabic-Greek astronomical tradition (likely Zij-based, 9th-12th c. Arabic sources). Entirely different from Ptolemaic Handy Tables. Models cannot predict values.
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O
   → Requires new training data or general HTR (Kraken/Transkribus)

🔴  Vat.gr.1047  [LOW]
   Table types: persian_syntaxis_chrysokokkes
   Reason: Persian Syntaxis: same remarks as Vat.gr.1058. Signum T in Bardi stemma.
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O
   → Requires new training data or general HTR (Kraken/Transkribus)

Overall Summary
===============
Total manuscripts in corpus:  12
  Models directly applicable: 7  (Handy Tables / Handy Tables–derived)
  Models NOT applicable:      5  (Persian/Arabic/other table types)

High-confidence manuscripts (4):
  Vat.gr.1291
  Vat.gr.1594
  Vat.gr.208
  Vat.gr.184

Medium-confidence manuscripts (3):
  Vat.gr.211  (adapted/partial Handy Tables)
  Vat.gr.198  (adapted/partial Handy Tables)
  Vat.pal.gr.278  (adapted/partial Handy Tables)

Per-cell breakdown for HIGH-confidence manuscripts (ensemble model):
  Note: these are estimates based on training RMSEs, not image-verified.
  Vat.gr.1291            TRANSCRIBABLE=100.0%  UNCERTAIN=0.0%  OFF_TABLE=0.0%
  Vat.gr.1594            TRANSCRIBABLE=100.0%  UNCERTAIN=0.0%  OFF_TABLE=0.0%
  Vat.gr.208             TRANSCRIBABLE=100.0%  UNCERTAIN=0.0%  OFF_TABLE=0.0%
  Vat.gr.184             TRANSCRIBABLE=100.0%  UNCERTAIN=0.0%  OFF_TABLE=0.0%

Interpretation:
  TRANSCRIBABLE % = % of cells where model prediction is reliable enough
                    to use as a starting transcription without manual check
  UNCERTAIN %     = % of cells where model gives a useful hint but
                    manual verification is recommended
  OFF TABLE %     = % of cells where model prediction is not useful

Important caveats:
  1. These estimates assume the target manuscript has the SAME table type
     and SAME klima (Klima V, Byzantium) as the training data.
  2. For other klimata in the same manuscript: anaph values will differ
     (need klima-specific model), but hour/chron are NOT present in HT oblique
     ascension tables for other klimata without retraining.
  3. The 'flat' model is better for anaph (monotone linear function).
     The 'topo' model is better for hour + chron (periodic on S¹).
  4. Manual verification by the user remains essential for all predictions.