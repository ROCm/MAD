# Report templates

The tooling generates one report per phase of one job. Anything that spans jobs is written
by hand, and these templates carry the structure and the disclaimers that such a document
needs so that the next reader can tell what was measured.

| template | when to use it |
|---|---|
| `model_comparison.template.md` | comparing models, configurations or clusters across several jobs |
| `run_findings.template.md` | one campaign's investigation: what was tried, what failed and why, what the numbers were |

Both are filled by copying into the run directory's `reports/` and replacing every
`<FILL_...>`. A placeholder left in place is the point: it shows what has not been
established yet.

Two rules that apply to every hand-written report here:

- **Every figure names its job.** A number without a job id cannot be checked, and
  profiled and unprofiled runs of the same manifest differ by several times.
- **The scope section is not optional.** These documents outlive the session that produced
  them, and the limits of the method are what stops them being quoted as something they are
  not. `references/interpretation.md` is the source for that section.
