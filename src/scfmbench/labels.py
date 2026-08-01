"""Cell-type label harmonisation: TISCH2 author labels -> Cell Ontology -> working class.

Rules, in order of precedence:
  1. An author label with an explicit mapping becomes its working class.
  2. An author label matching a DROP rule is dropped, and counted.
  3. An author label with NO rule at all raises. It is never guessed, never
     bucketed into "other", and never silently discarded -- an unrecognised
     vocabulary means the mapping table is out of date with the corpus, which is
     a data problem the analyst must see.

The distinction that matters scientifically: dropping an *ambiguous* label
("Immune cells", "Myeloid") is correct because assigning it to one working class
would inject label noise into exactly the comparison being measured. Dropping an
*unrecognised* label would hide a corpus change. So the first is a counted rule
and the second is an exception.
"""
from __future__ import annotations

import pandas as pd

# author label -> (working class, Cell Ontology term)
MAPPING: dict[str, tuple[str, str]] = {
    # --- T / NK ---
    "CD4Tconv":     ("CD4_T",       "CL:0000624"),
    "CD4T":         ("CD4_T",       "CL:0000624"),
    "Tfh":          ("CD4_T",       "CL:0002038"),
    "Th1":          ("CD4_T",       "CL:0000545"),
    "Th17":         ("CD4_T",       "CL:0000899"),
    "CD8T":         ("CD8_T",       "CL:0000625"),
    "CD8Tex":       ("CD8_T",       "CL:0000625"),
    "Tex":          ("CD8_T",       "CL:0000625"),
    "Treg":         ("Treg",        "CL:0000815"),
    "NK":           ("NK",          "CL:0000623"),
    "NKT":          ("NK",          "CL:0000814"),
    # --- B / plasma ---
    "B":            ("B",           "CL:0000236"),
    "Plasma":       ("Plasma",      "CL:0000786"),
    "PlasmaB":      ("Plasma",      "CL:0000786"),
    # --- myeloid ---
    "Mono/Macro":   ("Mono_Macro",  "CL:0000576"),
    "Monocyte":     ("Mono_Macro",  "CL:0000576"),
    "Macrophage":   ("Mono_Macro",  "CL:0000235"),
    "DC":           ("DC",          "CL:0000451"),
    "pDC":          ("DC",          "CL:0000784"),
    "cDC1":         ("DC",          "CL:0000990"),
    "cDC2":         ("DC",          "CL:0002399"),
    "Mast":         ("Mast",        "CL:0000097"),
    "Neutrophils":  ("Neutrophil",  "CL:0000775"),
    "Neutrophil":   ("Neutrophil",  "CL:0000775"),
    # --- non-immune context classes (retained: they are the confusable background) ---
    "Malignant":    ("Malignant",   "CL:0001064"),
    "Epithelial":   ("Epithelial",  "CL:0000066"),
    "Fibroblasts":  ("Fibroblast",  "CL:0000057"),
    "Fibroblast":   ("Fibroblast",  "CL:0000057"),
    "Myofibroblast":("Fibroblast",  "CL:0000186"),
    "Myofibroblasts":("Fibroblast", "CL:0000186"),
    "Endothelial":  ("Endothelial", "CL:0000115"),
}

# Labels dropped deliberately, with the scientific reason. Counted in T4.
DROP_RULES: dict[str, str] = {
    "Immune cells":  "granularity too coarse -- spans multiple working classes",
    "Lymphocyte":    "granularity too coarse -- spans T, B and NK",
    "Myeloid":       "granularity too coarse -- spans monocyte/macrophage, DC, mast, neutrophil",
    "Tcell":         "granularity too coarse -- spans CD4, CD8 and Treg",
    "TNK":           "granularity too coarse -- spans T and NK",
    "Others":        "unspecified by the original authors",
    "Other":         "unspecified by the original authors",
    "Unknown":       "unspecified by the original authors",
    "Unclassified":  "unspecified by the original authors",
    "Doublet":       "technical artefact",
    "Doublets":      "technical artefact",
    "Low quality":   "technical artefact",
    "Tprolif":       "proliferating compartment spans multiple identities",
    "Proliferating": "proliferating compartment spans multiple identities",
    "Cycling":       "proliferating compartment spans multiple identities",
    # Tissue-resident non-TME populations: present in a few datasets (e.g. brain
    # metastasis) but not part of the tumour-immune question and too rare across
    # the corpus to support a class.
    "Oligodendrocyte": "non-TME resident cell type, not part of the immune question",
    "Pericytes":       "non-TME resident cell type, not part of the immune question",
    "Astrocyte":       "non-TME resident cell type, not part of the immune question",
    "Hepatocyte":      "tissue parenchyma, not part of the immune question",
    "Erythrocytes":    "not a nucleated immune population of interest",
    "Acinar":          "tissue parenchyma, not part of the immune question",
    "Ductal":          "tissue parenchyma, not part of the immune question",
    "Endocrine":       "tissue parenchyma, not part of the immune question",
    "Stellate":        "tissue parenchyma, not part of the immune question",
    "Enteric glia":    "tissue parenchyma, not part of the immune question",
    "Secretory":       "tissue parenchyma, not part of the immune question",
    "Progenitor":      "developmental state spans multiple identities",
    "Stem":            "developmental state spans multiple identities",
    # --- rules added after auditing the ACTUAL corpus vocabulary (29 labels) ---
    # "Tproilf" is a misspelling of "Tprolif" present in the source metainfo of one
    # dataset (330 cells). It is given its OWN explicit rule rather than being
    # normalised by fuzzy matching: a typo in upstream data is a fact about the data,
    # and silently correcting it would hide that the vocabulary is not clean.
    "Tproilf":         "proliferating compartment spans multiple identities "
                       "(source-data misspelling of 'Tprolif'; mapped by an explicit rule, "
                       "not by fuzzy matching)",
    "Pit mucous":      "gastric surface epithelium -- tissue parenchyma, not part of the immune question",
    "Gland mucous":    "gastric gland epithelium -- tissue parenchyma, not part of the immune question",
    "SMC":             "smooth muscle -- structural stroma, not part of the immune question",
    "Myocyte":         "muscle -- structural stroma, not part of the immune question",
    "Hepatic progenitor": "tissue parenchyma progenitor, not part of the immune question",
    # ILC (innate lymphoid cells) ARE immune, but appear in a single dataset at 202
    # cells corpus-wide. They cannot support a class in a cross-dataset design (they
    # would be absent from 12 of 13 datasets and from most training partitions), and
    # folding them into NK would assert a lineage equivalence the data does not
    # establish. Dropped as too rare to model, and the count is reported.
    "ILC":             "innate lymphoid cells: immune, but 202 cells in one dataset only -- "
                       "too rare to form a class in a cross-dataset design, and not "
                       "equivalent to NK. Dropped and counted rather than merged.",
}

IMMUNE_CLASSES = ("CD4_T", "CD8_T", "Treg", "NK", "B", "Plasma",
                  "Mono_Macro", "DC", "Mast", "Neutrophil")


class UnmappedLabelError(ValueError):
    """An author label has neither a mapping nor a drop rule. Fail loudly."""


def harmonise_labels(labels: pd.Series) -> pd.DataFrame:
    """Map author labels to working classes.

    Returns a frame with `working_class` (None where dropped), `cl_term`, and
    `drop_reason`. Raises on any label the mapping table has never seen.
    """
    seen = pd.Index(labels.astype(str).unique())
    unknown = [s for s in seen if s not in MAPPING and s not in DROP_RULES]
    if unknown:
        raise UnmappedLabelError(
            f"{len(unknown)} author label(s) have no mapping and no drop rule: "
            f"{sorted(unknown)}. Add them to MAPPING or DROP_RULES in labels.py -- "
            f"they must not be guessed, bucketed into 'other', or silently dropped."
        )
    s = labels.astype(str)
    return pd.DataFrame({
        "label_raw": s,
        "working_class": s.map(lambda x: MAPPING[x][0] if x in MAPPING else None),
        "cl_term": s.map(lambda x: MAPPING[x][1] if x in MAPPING else None),
        "drop_reason": s.map(lambda x: DROP_RULES.get(x)),
    }, index=labels.index)


def mapping_table() -> pd.DataFrame:
    """Table T4 skeleton: the full mapping, independent of any dataset."""
    rows = [{"label_raw": k, "cl_term": v[1], "working_class": v[0], "action": "map",
             "reason": ""} for k, v in sorted(MAPPING.items())]
    rows += [{"label_raw": k, "cl_term": "", "working_class": "", "action": "drop",
              "reason": v} for k, v in sorted(DROP_RULES.items())]
    return pd.DataFrame(rows)
