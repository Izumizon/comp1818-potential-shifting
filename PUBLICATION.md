# Remaining external publication step

The connected GitHub capabilities available in this session did not provide a
new-repository creation action. No hosting account, repository or dataset URL
was invented or published, and no Moodle submission was made.

1. Upload the complete reproducibility archive to a location permitted by the
   teaching team (for example, an appropriate repository or shared drive).
2. Ensure the implementation, environment dataset, configs and results are
   accessible to the teaching team. Test permissions using viewer access rather
   than relying on the owner's logged-in session.
3. Put the genuine implementation and dataset/artefact URLs into
   `report/publication.json`, then rerun `build_report.py` and compile the PDF.
   The same verified shared archive may contain both, if clearly labelled.
4. Run the final report build with `--require-published`; this rejects blank
   URLs. Inspect the final PDF, link access, anonymity, AI declaration and receipt
   rules before uploading that single PDF to Moodle.

Current report text explicitly says publication is pending. It must not be
treated as a submission-ready PDF until real links have been inserted and
verified. Scope approval and confirmation of the user-reported AI permission
also remain external actions; neither is silently assumed completed.
