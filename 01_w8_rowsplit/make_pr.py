#!/usr/bin/env python3
"""Assemble and file the W8 row-split weight-decode submission, one explicit stage at a time.

The point of the staging is that the text lives in files next to this script and can be re-read
and edited between stages. Nothing here touches GitHub unless a stage that writes is named on the
command line; the default stage only reports.

Stages
  check       validate the files and show what every later stage would do. Changes nothing.
  issue       open the Issue from ISSUE_TITLE.txt and ISSUE_BODY.md.
  bundle      pull the branch off the build server as a git bundle (ssh + scp).
  fetch       import that bundle into a local work clone under ./work.
  push        push the branch to the fork. Rewrites the remote branch.
  create      open the pull request from PR_TITLE.txt and PR_BODY.md.

Typical order:  check -> issue -> (wait for the maintainer to confirm) -> bundle -> fetch
                -> push -> create

CONTRIBUTING.md now requires a confirmed Issue before a pull request, so `issue` comes first and
the branch does not go anywhere until he answers. Every writing stage prints exactly what it is
about to do and needs --yes to proceed.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE_FILE = HERE / "PR_TITLE.txt"
BODY_FILE = HERE / "PR_BODY.md"
ISSUE_TITLE_FILE = HERE / "ISSUE_TITLE.txt"
ISSUE_BODY_FILE = HERE / "ISSUE_BODY.md"
WORK = HERE / "work"
BUNDLE = HERE / "branch.bundle"

UPSTREAM = "Neroued/ninfer"
FORK = "MichaelDementii/ninfer"
BRANCH = "perf/widen-w8-rowsplit-decode"
REMOTE_BRANCH = "perf/widen-w8-rowsplit-decode"
# The branch is rebased onto current origin/master, so the PR targets master directly.
BASE_BRANCH = "master"

KEY = "C:/Users/MixaPC/.ssh/ninfer_5090_ed25519"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
       "-i", KEY, "-p", "28677", "root@180.189.55.43"]
SCP_HOST = "root@180.189.55.43"
SERVER_REPO = "/root/ninfer_d4"


def run(cmd, cwd=None, check=True, quiet=False):
    if not quiet:
        print("  $", " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0 and check:
        print((p.stdout or "") + (p.stderr or ""))
        sys.exit(f"failed: {' '.join(str(c) for c in cmd)}")
    return (p.stdout or "").strip()


def read_texts(title_file, body_file):
    if not title_file.exists() or not body_file.exists():
        sys.exit(f"missing {title_file.name} or {body_file.name}")
    return title_file.read_text(encoding="utf-8").strip(), body_file.read_text(encoding="utf-8")


def validate(title, body):
    problems = []
    if not title:
        problems.append("title is empty")
    if len(title) > 120:
        problems.append(f"title is {len(title)} chars, over the 120 the UI truncates at")
    if "\n" in title:
        problems.append("title spans more than one line")
    if len(body.strip()) < 500:
        problems.append("body is suspiciously short")
    for token in ("TODO", "TBD", "XXX", "<<", "PLACEHOLDER"):
        if token in body:
            problems.append(f"body still contains {token}")
    for n, line in enumerate(body.splitlines(), 1):
        if re.search(r"\|\s*\d+\.\d+%\s*\|", line) and "±" not in line:
            problems.append(f"line {n}: percentage without a sign, is it a gain or a loss?")
    return problems


def stage_check(args):
    problems = []
    for label, tf, bf in (("PR", TITLE_FILE, BODY_FILE),
                          ("Issue", ISSUE_TITLE_FILE, ISSUE_BODY_FILE)):
        if not tf.exists() or not bf.exists():
            problems.append(f"{label}: missing {tf.name} or {bf.name}")
            continue
        title, body = read_texts(tf, bf)
        print(f"{label} title ({len(title)} chars):\n  {title}")
        print(f"{label} body: {len(body.splitlines())} lines, {len(body)} chars")
        for h in [l for l in body.splitlines() if l.startswith("#")]:
            print("   ", h)
        problems += [f"{label}: {p}" for p in validate(title, body)]
        print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("no problems found")
    print(f"\nwould open  Issue on {UPSTREAM}")
    print(f"would push  {BRANCH}  ->  {FORK}:{REMOTE_BRANCH}")
    print(f"would open  PR against {UPSTREAM}:{BASE_BRANCH}")
    return 1 if problems else 0


def stage_issue(args):
    title, body = read_texts(ISSUE_TITLE_FILE, ISSUE_BODY_FILE)
    problems = validate(title, body)
    if problems:
        print("refusing to open an Issue with unresolved problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"opening Issue on {UPSTREAM}")
    print(f"  title: {title}")
    print(f"  body:  {len(body.splitlines())} lines")
    if not args.yes:
        return print("dry run; pass --yes")
    tmp = HERE / ".issue.tmp"
    tmp.write_text(body, encoding="utf-8")
    url = run(["gh", "issue", "create", "--repo", UPSTREAM, "--title", title,
               "--body-file", str(tmp)])
    tmp.unlink(missing_ok=True)
    print(url)
    (HERE / "ISSUE_URL.txt").write_text(url + "\n", encoding="utf-8")


def stage_bundle(args):
    print(f"pulling {BRANCH} off the build server as a bundle")
    if not args.yes:
        return print("dry run; pass --yes")
    run(SSH + [f"cd {SERVER_REPO} && git bundle create /tmp/w8.bundle "
               f"origin/{BASE_BRANCH}..{BRANCH} --branches={BRANCH} 2>&1 | tail -3"])
    run(["scp", "-q", "-o", "BatchMode=yes", "-i", KEY, "-P", "28677",
         f"{SCP_HOST}:/tmp/w8.bundle", str(BUNDLE)])
    print(f"bundle: {BUNDLE} ({BUNDLE.stat().st_size} bytes)")


def stage_fetch(args):
    if not BUNDLE.exists():
        sys.exit("no bundle; run the bundle stage first")
    print(f"importing the bundle into {WORK}")
    if not args.yes:
        return print("dry run; pass --yes")
    if not WORK.exists():
        run(["git", "clone", "--quiet", f"https://github.com/{UPSTREAM}.git", str(WORK)])
        run(["git", "remote", "add", "fork", f"https://github.com/{FORK}.git"], cwd=WORK)
    run(["git", "fetch", "--quiet", "origin"], cwd=WORK)
    run(["git", "fetch", str(BUNDLE), f"{BRANCH}:{BRANCH}", "--force"], cwd=WORK)
    n = run(["git", "rev-list", "--count", f"origin/{BASE_BRANCH}..{BRANCH}"], cwd=WORK, quiet=True)
    print(f"{BRANCH} imported: {n} commits ahead of origin/{BASE_BRANCH}")
    print(run(["git", "log", "--oneline", "-3", BRANCH], cwd=WORK, quiet=True))


def stage_push(args):
    if not WORK.exists():
        sys.exit("no work clone; run the fetch stage first")
    print(f"pushing {BRANCH} -> {FORK}:{REMOTE_BRANCH} (force-with-lease)")
    if not args.yes:
        return print("dry run; pass --yes")
    run(["git", "push", "--force-with-lease", "fork", f"{BRANCH}:{REMOTE_BRANCH}"], cwd=WORK)


def stage_create(args):
    title, body = read_texts(TITLE_FILE, BODY_FILE)
    problems = validate(title, body)
    issue_url = HERE / "ISSUE_URL.txt"
    if not issue_url.exists():
        problems.append("no ISSUE_URL.txt; CONTRIBUTING.md wants a linked, confirmed Issue. "
                        "Run the issue stage, wait for his answer, then come back.")
    else:
        number = issue_url.read_text(encoding="utf-8").strip().rsplit("/", 1)[-1]
        if not number.isdigit():
            problems.append(f"ISSUE_URL.txt does not end in an issue number: {number}")
        elif "#ISSUE" in body:
            body = body.replace("#ISSUE", f"#{number}")
            print(f"  substituted #ISSUE -> #{number}")
        elif f"#{number}" not in body:
            problems.append(f"the PR body mentions neither #ISSUE nor #{number}")
    if problems:
        print("refusing to open a PR with unresolved problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"opening PR on {UPSTREAM}, base {BASE_BRANCH}, head {FORK.split('/')[0]}:{REMOTE_BRANCH}")
    print(f"  title: {title}")
    if not args.yes:
        return print("dry run; pass --yes")
    tmp = HERE / ".body.tmp"
    tmp.write_text(body, encoding="utf-8")
    url = run(["gh", "pr", "create", "--repo", UPSTREAM, "--base", BASE_BRANCH,
               "--head", f"{FORK.split('/')[0]}:{REMOTE_BRANCH}",
               "--title", title, "--body-file", str(tmp)])
    tmp.unlink(missing_ok=True)
    print(url)
    (HERE / "PR_URL.txt").write_text(url + "\n", encoding="utf-8")


STAGES = {"check": stage_check, "issue": stage_issue, "bundle": stage_bundle,
          "fetch": stage_fetch, "push": stage_push, "create": stage_create}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="check", choices=sorted(STAGES))
    ap.add_argument("--yes", action="store_true", help="actually perform a writing stage")
    a = ap.parse_args()
    sys.exit(STAGES[a.stage](a) or 0)
