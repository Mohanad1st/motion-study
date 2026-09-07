# Security and privacy reporting

## Reporting

Please do not open a public issue for a security or privacy problem.

Use GitHub's private vulnerability reporting on this repository
(**Security → Report a vulnerability**), which reaches the maintainer without disclosing the report.

Expect an acknowledgement within seven days. This is maintained by one person, not a product with
an on-call rotation — please size your expectations accordingly, and say in the report if you intend
to disclose publicly on a deadline.

## The thing most worth reporting

This software processes **video of identifiable people at work**. The privacy boundary matters more
here than any conventional vulnerability class, because a leak is not a database of hashes — it is
footage of named employees.

The design intent is that footage and analysis outputs never leave the machine that processes them:

- `input/` and `runs/` are gitignored, and `.gitignore` has excluded them since the first commit;
- nothing is uploaded, and no network call is made during analysis;
- model weights are downloaded once from the `rtmlib` project and cached locally;
- `report.html` is self-contained and opens offline.

**Anything that breaks one of those is a genuine finding**, and the most valuable report you can
make. Concretely: a path by which a frame, a track, a worker name, or a report reaches the network;
a way a generated report embeds a source frame you would not expect; a default that writes footage
somewhere outside `input/` or `runs/`.

## Also in scope

- A credential, key, or token committed anywhere in the git history.
- A crafted video or config file that leads to code execution rather than a clean error.
- Path traversal via a config value — station names and paths from `configs/*.yaml` reach the
  filesystem and the generated HTML.
- HTML or script injection into `report.html` through a config value, worker name, or station label,
  since the report is rendered from a Jinja template and opened in a browser.

## Not in scope

- Detection or tracking inaccuracy. Real-footage detection accuracy is explicitly **not validated**
  — the README says so. A missed worker is a known limitation, not a vulnerability.
- Dependency advisories with no exploit path in this usage.
- The absence of authentication. There is no server; this is a local command-line tool.

## A note on lawful use

Filming people at work is regulated differently in different places, and consent, notice and
works-council obligations are not optional. `docs/RUNBOOK.md` covers the practical side. If you find
that this tool makes it *easier* to do something unlawful or covert — for example a default that
hides the fact that recording is happening — please report that too. It is a design defect.
