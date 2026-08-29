#!/usr/bin/env python3
"""Assemble and file the W8 row-split weight-decode submission, one explicit stage at a time.

The text lives in files next to this script so it can be re-read and edited between stages.
Nothing here touches GitHub unless a stage that writes is named on the command line; the default
stage only reports.

Stages
  check       validate the files and show what every later stage would do. Changes nothing.
  issue       open the Issue from ISSUE_TITLE.txt and ISSUE_BODY.md.
  bundle      pull the branch off the build server as a git bundle (ssh + scp).
  fetch       import that bundle into a local work clone under ./work.
  push        push the branch to the fork. Rewrites the remote branch.
  create      open the pull request from PR_TITLE.txt and PR_BODY.md.

Typical order:  check -> issue -> (wait for him to confirm) -> bundle -> fetch -> push -> create

CONTRIBUTING.md requires a confirmed Issue before a pull request, so `issue` comes first and the
branch does not go anywhere until he answers. `create` enforces that with two gates it can actually
check: ISSUE_CONFIRMED.txt must exist and say what he agreed to, and the branch must still be
exactly one commit on top of the current origin/master. Every writing stage prints what it is about
to do and needs --yes.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE_FILE = HERE / "PR_TITLE.txt"
BODY_FILE = HERE / "PR_BODY.md"
ISSUE_TITLE_FILE = HERE / "ISSUE_TITLE.txt"
ISSUE_BODY_FILE = HERE / "ISSUE_BODY.md"
ISSUE_URL_FILE = HERE / "ISSUE_URL.txt"
CONFIRMED_FILE = HERE / "ISSUE_CONFIRMED.txt"
WORK = HERE / "work"
BUNDLE = HERE / "branch.bundle"

UPSTREAM = "Neroued/ninfer"
FORK = "MichaelDementii/ninfer"
BRANCH = "perf/widen-w8-rowsplit-decode"
REMOTE_BRANCH = "perf/widen-w8-rowsplit-decode"
BASE_BRANCH = "master"

# The key lives in a different place on each of our machines; NINFER_KEY overrides.
KEY_CANDIDATES = [
    os.environ.get("NINFER_KEY", ""),
    str(Path.home() / ".ssh_ninfer" / "key"),
    "C:/Users/MixaPC/.ssh/ninfer_5090_ed25519",
    str(Path.home() / ".ssh" / "ninfer_5090_ed25519"),
    str(HERE.parent / "vast_47972656"),
]
SCP_HOST = "root@180.189.55.43"
PORT = "28677"
SERVER_REPO = "/root/ninfer_d4"


def key_path():
    for k in KEY_CANDIDATES:
        if k and Path(k).exists():
            return k
    sys.exit("no ssh key found; set NINFER_KEY to its path")


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


def validate(title, body, kind):
    """Only checks that can actually be wrong. An earlier version flagged every percentage in a
    table cell as unsigned, which caught eleven legitimate share and spread cells and refused to
    file anything; a validator that cries wolf is worse than none."""
    problems = []
    if not title:
        problems.append("title is empty")
    if len(title) > 120:
        problems.append(f"title is {len(title)} chars, over the 120 the UI truncates at")
    if "\n" in title:
        problems.append("title spans more than one line")
    if len(body.strip()) < 500:
        problems.append("body is suspiciously short")
    for token in ("TODO", "TBD", "XXX", "PLACEHOLDER", "FIXME"):
        if token in body:
            problems.append(f"body still contains {token}")
    if kind == "pr":
        if "#ISSUE" not in body and not re.search(r"#\d+", body):
            problems.append("the PR body references no Issue at all")
        if "Scope as confirmed" in body and "…" in body.split("Scope as confirmed", 1)[1][:200]:
            problems.append("the agreed-scope line is still a placeholder")
    return problems


def stage_check(args):
    problems = []
    for kind, label, tf, bf in (("issue", "Issue", ISSUE_TITLE_FILE, ISSUE_BODY_FILE),
                                ("pr", "PR", TITLE_FILE, BODY_FILE)):
        if not tf.exists() or not bf.exists():
            problems.append(f"{label}: missing {tf.name} or {bf.name}")
            continue
        title, body = read_texts(tf, bf)
        print(f"{label} title ({len(title)} chars):\n  {title}")
        print(f"{label} body: {len(body.splitlines())} lines, {len(body)} chars")
        for h in [l for l in body.splitlines() if l.startswith("#")]:
            print("   ", h)
        problems += [f"{label}: {p}" for p in validate(title, body, kind)]
        print()
    print("Issue filed:      ", ISSUE_URL_FILE.read_text().strip() if ISSUE_URL_FILE.exists() else "not yet")
    print("Scope confirmed:  ", "yes" if CONFIRMED_FILE.exists() else "no - create will refuse")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nno problems found")
    print(f"\nwould open  Issue on {UPSTREAM}")
    print(f"would push  {BRANCH}  ->  {FORK}:{REMOTE_BRANCH}")
    print(f"would open  PR against {UPSTREAM}:{BASE_BRANCH}")
    return 1 if problems else 0


def stage_issue(args):
    title, body = read_texts(ISSUE_TITLE_FILE, ISSUE_BODY_FILE)
    problems = validate(title, body, "issue")
    if problems:
        print("refusing to open an Issue with unresolved problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    if ISSUE_URL_FILE.exists():
        sys.exit(f"ISSUE_URL.txt already exists ({ISSUE_URL_FILE.read_text().strip()}); "
                 "delete it if you really mean to open a second Issue")
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
    ISSUE_URL_FILE.write_text(url + "\n", encoding="utf-8")
    print("\nNow wait for him. When he confirms the scope, write what he agreed to into\n"
          "ISSUE_CONFIRMED.txt - create refuses without it.")


def stage_bundle(args):
    key = key_path()
    print(f"pulling {BRANCH} off the build server as a bundle, key {key}")
    if not args.yes:
        return print("dry run; pass --yes")
    # No pipe: a pipeline would hand back tail's exit status and a failed bundle would go unnoticed,
    # leaving a stale /tmp/w8.bundle to be shipped.
    run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-i", key, "-p", PORT, SCP_HOST,
         f"rm -f /tmp/w8.bundle && cd {SERVER_REPO} && "
         f"git bundle create /tmp/w8.bundle origin/{BASE_BRANCH}..{BRANCH} --branches={BRANCH}"])
    BUNDLE.unlink(missing_ok=True)
    run(["scp", "-q", "-o", "BatchMode=yes", "-i", key, "-P", PORT,
         f"{SCP_HOST}:/tmp/w8.bundle", str(BUNDLE)])
    size = BUNDLE.stat().st_size
    if size < 512:
        sys.exit(f"bundle is only {size} bytes; that is not a real bundle")
    print(f"bundle: {BUNDLE} ({size} bytes)")


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
    run(["git", "fetch", "--quiet", "fork"], cwd=WORK, check=False)
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
    # force-with-lease needs a remote-tracking ref to compare against, and a fresh clone has none.
    run(["git", "fetch", "fork"], cwd=WORK, check=False)
    run(["git", "push", "--force-with-lease", "fork", f"{BRANCH}:{REMOTE_BRANCH}"], cwd=WORK)


def stage_create(args):
    title, body = read_texts(TITLE_FILE, BODY_FILE)
    problems = validate(title, body, "pr")

    if not ISSUE_URL_FILE.exists():
        problems.append("no ISSUE_URL.txt; run the issue stage and wait for his answer")
    if not CONFIRMED_FILE.exists():
        problems.append("no ISSUE_CONFIRMED.txt; CONTRIBUTING wants the agreed scope in the PR, "
                        "and this is the only gate that can check he actually answered")
    if not WORK.exists():
        problems.append("no work clone; run fetch first")

    number = None
    if ISSUE_URL_FILE.exists():
        number = ISSUE_URL_FILE.read_text(encoding="utf-8").strip().rsplit("/", 1)[-1]
        if not number.isdigit():
            problems.append(f"ISSUE_URL.txt does not end in an issue number: {number}")

    if WORK.exists():
        run(["git", "fetch", "--quiet", "origin"], cwd=WORK, quiet=True)
        ahead = run(["git", "rev-list", "--count", f"origin/{BASE_BRANCH}..{BRANCH}"],
                    cwd=WORK, quiet=True)
        behind = run(["git", "rev-list", "--count", f"{BRANCH}..origin/{BASE_BRANCH}"],
                     cwd=WORK, quiet=True)
        if ahead != "1":
            problems.append(f"branch is {ahead} commits ahead of {BASE_BRANCH}; he asks for one")
        if behind != "0":
            problems.append(f"{BASE_BRANCH} has moved {behind} commits ahead of the branch; "
                            "rebase on the server and re-run bundle/fetch/push")

    if problems:
        print("refusing to open a PR with unresolved problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    if "#ISSUE" in body:
        body = body.replace("#ISSUE", f"#{number}")
        print(f"  substituted #ISSUE -> #{number}")
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
