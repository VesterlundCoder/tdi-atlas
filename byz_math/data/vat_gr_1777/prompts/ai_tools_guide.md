# AI Tools Guide — Vat.gr.1777 Transcription Pipeline

---

## Summary: Which Tool for Which Task

| Task | Best Tool | Why |
|------|-----------|-----|
| Reading a manuscript line image | **GPT-4o (vision)** | Best vision + Greek Unicode output |
| Verifying / second opinion | **Claude 3.5 Sonnet/Opus** | Lower hallucination rate on uncertain text |
| Batch processing long sessions | **Gemini 1.5 Pro** | 1M token context; track many folios |
| Semi-automatic line segmentation | **Kraken** (CLI) | Built for historical MSS layout analysis |
| HTR training (Phase 2+) | **Kraken** or **Transkribus** | Mature Byzantine Greek models available |
| Collaborative annotation | **eScriptorium** | Web UI, integrates Kraken, free/open |
| Searching transcription patterns | **Logion** (Princeton) | Greek-specific BERT; lacuna completion |

---

## Tool 1 — GPT-4o (Primary Transcription Assistant)

**URL:** https://chat.openai.com → Custom GPT (use `gpt_system_prompt.md`)  
**Model:** GPT-4o (not GPT-4-turbo — need vision)

### Setup
1. Go to ChatGPT → Explore GPTs → Create
2. Paste the entire content of `prompts/gpt_system_prompt.md` as the System Instructions
3. Upload as knowledge files:
   - `rules/unicode_reference.md`
   - `rules/transcription_rules.md`
   - `rules/abbreviations_catalog.md`
   - `glyph_inventory/glyph_inventory.csv`
4. Name the GPT: **ByzPaleo-1777**

### Workflow per line
```
1. Open 0007_1r.jpg (or crop)
2. Upload image to Custom GPT
3. Prompt: "Provide DIPLOMATIC transcription of this manuscript line.
            Mark all uncertainty per transcription_rules.md.
            Use lunate sigma ϲ where appropriate for this 15th-c. hand."
4. Review output character by character
5. Correct errors — save to diplomatic/0007_1r_dipl.txt
```

### What GPT-4o does well
- Recognising clearly-written Byzantine minuscule letters
- Detecting abbreviation marks
- Outputting polytonic Unicode directly
- Noticing text structure (main text vs. margin)

### What GPT-4o does poorly
- Ambiguous/faded glyphs (it guesses rather than marks [?])
- Distinguishing ν/υ, ο/σ in degraded ink
- Consistency across sessions (no memory between chats unless Custom GPT)
- Mathematical/logical notation it hasn't seen before

**Critical rule:** Always override GPT's "confident wrong reading" with your own judgment.

---

## Tool 2 — Claude (Verification & Uncertainty Checking)

**URL:** https://claude.ai (Claude 3.5 Sonnet or Opus)

Use Claude as a **second reader** when GPT-4o produces uncertain output.

### Prompt template for verification
```
I am working on a diplomatic transcription of Vat.gr.1777, a Byzantine Greek 
manuscript (mid-15th c., scholarly minuscule).

GPT-4o read this line as: [GPT output]

Here is the line image: [attach image]

Please:
1. Verify each character against the image
2. Mark any characters you read differently
3. Mark any characters you are uncertain about as [x?]
4. Do NOT normalize, correct, or expand abbreviations
5. Only transcribe what is visually present
```

### Claude's comparative advantages
- More likely to express genuine uncertainty
- Better at explaining paleographic reasoning
- Useful for discussing whether a glyph matches known letterforms
- Good for checking if a word makes grammatical sense **without** normalising it

---

## Tool 3 — Kraken (Line Segmentation + HTR Training)

**URL:** https://kraken.re  
**Install:** `pip install kraken`

### Phase 1 use: Layout analysis (line segmentation)

Kraken can automatically segment manuscript pages into lines — saving the manual cropping step.

```bash
# Install
pip install kraken

# Segment a folio image into lines
kraken -i 0007_1r.jpg lines.json segment -bl

# Output: lines.json with bounding polygons for each line
# Use these to crop line images automatically
```

### Phase 2 use: HTR training (after ground truth is ready)

Once you have 50+ aligned pairs of (line image, diplomatic text):

```bash
# Prepare ground truth (eScriptorium export format or manual)
# Each pair: image.png + image.gt.txt (the diplomatic transcription)

# Train a model
ketos train -f alto ground_truth/*.xml

# Transcribe with trained model
kraken -i new_folio.jpg output.txt ocr -m my_model.mlmodel
```

**Pre-trained Byzantine model:** The **Bodleian Greek HTR model** (from SunoikisisDC 2023-24) is available for download and provides a starting point. Check: https://github.com/SunoikisisDC/SunoikisisDC-2023-2024

---

## Tool 4 — eScriptorium (Annotation Platform)

**URL:** https://escriptorium.inria.fr (or self-host)  
**Alternative instance:** https://escriptorium.paris.fr

eScriptorium is a web-based annotation platform that integrates Kraken. It is the **recommended professional tool** for building HTR ground truth.

### What it does
- Upload manuscript images
- Automatic line segmentation (via Kraken)
- Side-by-side image / transcription editor
- Export to ALTO XML, PAGE XML, or plain text
- Built-in versioning

### Setup for this project
1. Register at https://escriptorium.paris.fr (free academic accounts)
2. Create a project: "Vat.gr.1777 Byzantine HTR"
3. Upload your 236 images (or start with the first 20 folios)
4. Run Kraken segmentation
5. Transcribe diplomatically in the text editor
6. Export for Kraken training

---

## Tool 5 — Transkribus (Industry Standard HTR)

**URL:** https://www.transkribus.org  
**Cost:** Free tier (500 credits/month); academic pricing available

Transkribus has **pre-trained Greek manuscript models** you can immediately fine-tune on your data.

### Available Greek models to start with
- Search "Greek" in Transkribus model marketplace
- Look for: "Greek Manuscripts 15th century" or "Byzantine minuscule"
- Most relevant: models trained on philosophical/theological Greek MSS of the same period

### Workflow
1. Upload Vat.gr.1777 images to Transkribus
2. Run automatic text recognition with the best Greek model as baseline
3. Correct the output → this IS your ground truth building
4. After 50–100 corrected pages: train a custom model (ATR)
5. The custom model gets better with each corrected page

---

## Tool 6 — Logion (Princeton NLP for Greek)

**URL:** https://www.logionproject.princeton.edu  
**Paper:** arXiv:2305.01099

Logion is a BERT model trained on the largest premodern Greek corpus to date. It can:
- Suggest completions for lacunae
- Detect scribal errors
- Propose textual emendations

**Use case for this project:**
- After building a diplomatic transcription, use Logion to flag statistically unusual character sequences (possible transcription errors)
- Do NOT use Logion to pre-fill gaps — use it only for post-hoc verification

```python
# Logion API (if available) or model from HuggingFace
# Model: check https://www.logionproject.princeton.edu for latest release
from transformers import pipeline
logion = pipeline("fill-mask", model="logion-project/greek-bert")
result = logion("τοῖς ζητοῦσι τὴν [MASK]")
```

---

## Phase Roadmap

### Phase 0 (Now — 2 weeks): Infrastructure & First Folio
- [x] Download 236 IIIF images ← DONE
- [x] Set up directory structure ← DONE
- [x] Build unicode_reference.md ← DONE
- [x] Build Custom GPT prompt ← DONE
- [ ] Set up eScriptorium account
- [ ] Transcribe ff. 1r–5v manually (folios 0007–0016) as calibration
- [ ] Build initial glyph inventory (50+ entries)

### Phase 1 (Weeks 3–8): First 20 Perfect Folios
- [ ] Diplomatic transcription: ff. 1r–20v (canvas 0007–0046)
- [ ] Expanded transcription for same
- [ ] glyph_inventory.csv: 200+ entries
- [ ] Argyros folios (0163–0164): special focus
- Target: 40 aligned pairs (image line ↔ diplomatic text)

### Phase 2 (Months 3–4): HTR Model Training
- [ ] Prepare Kraken ground truth format (50+ line pairs)
- [ ] Fine-tune on existing Byzantine Greek base model
- [ ] Evaluate on held-out folios
- [ ] Iterate

### Phase 3 (Months 5–6): Full Manuscript + Research Output
- [ ] Complete manuscript transcription (AI-assisted)
- [ ] TEI/XML encoding
- [ ] Argyros commentary: mathematical analysis
- [ ] Dataset release + paper draft
