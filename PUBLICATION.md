# Publication and access verification — 5 November 2026

The complete source, generated dataset and saved experimental records are in a
dedicated public repository:

- [Implementation, tests and saved results](https://github.com/Izumizon/comp1818-potential-shifting)
- [Generated dataset and its documentation](https://github.com/Izumizon/comp1818-potential-shifting/tree/main/data)

`report/publication.json` contains these genuine URLs. They were checked using
fresh standard-library HTTP openers with no GitHub credentials, Authorization
header, cookie jar or owner browser session. Repository metadata confirmed
public visibility; both linked pages returned HTTP 200. Selected published
source files and dataset arrays matched the local bytes exactly. Evidence is
retained in `verification/public_access_r9.json`; run `verify_publication.py`
to repeat the checks into a new output record.

The public source and dataset do not require the teaching team to be added as
collaborators. Access must nevertheless be rechecked near submission because
hosting availability or permissions can change.

## Final build and release procedure

1. Run the tests and `python build_report.py --require-published`.
2. Compile `report/main.tex` with the unchanged official class and style.
3. Verify no more than four counted pages, including the AI reflection; inspect
   every rendered page and both PDF link annotations.
4. Refresh the intentional delivery manifest, verify it, and package all sources,
   configurations, generated data, observations, models, figures and test records.
5. Publish the versioned release package and verify its download without owner
   authentication, including all embedded manifest hashes. The final readiness
   checklist and download-verification record accompany the versioned release.

No Moodle submission has been performed. The teaching-team AI-policy and
topic/extension questions remain unresolved; see
`verification/TEACHING_TEAM_QUESTIONS.md`. Public delivery and technical review
must not be misrepresented as institutional approval or submission readiness.
