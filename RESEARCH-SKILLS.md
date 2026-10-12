# Brain-tumour detection research skills

Status: selected and installer prepared; NOT downloaded or installed in this session.
Shell connections to GitHub were blocked. Global writes are outside the session's writable workspace. The official skill-installer Python helper could not run because no Python runtime is installed, so the fallback script uses Git and PowerShell.

Run from this project in a local PowerShell terminal:

```powershell
.\install-research-skills.ps1
```

This installs into `.agents/skills` for this project. To use your requested user-level location instead:

```powershell
.\install-research-skills.ps1 -Global
```

Preview destinations without downloading:

```powershell
.\install-research-skills.ps1 -CheckOnly
```

| Skills | Purpose |
| --- | --- |
| lit-search, medical-imaging-review | Find literature and compare medical-imaging methods with traceable sources |
| brainstorming-research-ideas | Develop testable architecture hypotheses and investigate novelty |
| pytorch-lightning | Organize training and reproducible model experiments |
| pydicom | Read DICOM imaging metadata if the dataset uses this format |
| statistical-analysis | Choose analyses and quantify uncertainty in comparisons |
| scientific-visualization | Produce figures from measured results |
| citation-management | Verify references and maintain BibTeX |
| research-paper, ml-paper-writing | Draft original research from completed experiments and verified evidence |
| peer-review | Critique methodology, evidence, and manuscript claims |
| latex-document | Typeset papers and supporting documents |
| stop-slop | Edit filler and formulaic prose while preserving technical meaning |

Sources: https://github.com/hardikpandya/stop-slop, https://github.com/luwill/research-skills, https://github.com/Orchestra-Research/AI-Research-SKILLs, https://github.com/K-Dense-AI/scientific-agent-skills, https://github.com/ndpvt-web/latex-document-skill. Selection also used the supplied PaperBanana pages and https://github.com/Epsilon617/Codex-Academic-Skills, which is an index rather than an installable skill bundle.

The script retains full supporting files, skips existing skills, records source commits for new installs, and does not execute downloaded scripts or install dependencies. Review the upstream instructions before first use. Python, uv, PyTorch and TeX dependencies must be provisioned separately as needed; some bundled scripts assume Bash and may require Git Bash or WSL on Windows. lit-search documents uv and Python 3.11+ as prerequisites.

For this project, distinguish tumour-presence classification, tumour-type classification, lesion localization, and segmentation before designing the architecture. Plan patient-level dataset splits and inspect duplicates before experiments. Compare with suitable baselines, run ablations, preserve seeds/configuration/results, and tie paper claims to recorded evidence. Treat architecture novelty as a claim to investigate, rather than an assumed outcome.

Writing assistance should support original reasoning, accurate paraphrasing, citations and disclosure required by the venue. No skill can guarantee an AI-detector or plagiarism-detector score. Stop Slop's style preferences should not damage precise scientific wording or override journal requirements.
