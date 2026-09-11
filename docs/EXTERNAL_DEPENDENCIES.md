# External dependencies

The CPU checks need Python, pytest and PyYAML only. Optional numeric figure reproduction needs NumPy, Matplotlib and system TeX packages (including mathpazo/Palatino support, type1cm/type1ec). The original thesis uses the bundled MastersDoctoralThesis class plus externally installed LaTeX packages; do not copy protected fonts.

Model IDs and commit revisions are identifiers, not included files. See MODEL_AND_REVISION_MANIFEST.md. VCR is external; see DATA_ACCESS.md. Original frozen execution manifests, checkpoints and generation/scoring jobs are documented through hash-bearing stubs. Projections in reports/frozen contain numeric results, not model-ready data or original rationale text.

The description-generation code contains the historical GPT-4o prompt/API settings. A mutable API model alias cannot guarantee the same outputs later. API steps require local credentials and may incur costs; CI never executes them. Public code licences remain pending; upstream dependencies retain their own terms.
