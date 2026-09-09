# Data and code access

The new GitHub directory is a **source-code release**, not a clinical dataset
release. Published aggregate tables and figures accompany the revised Reply
and Supplementary Information. This upload contains no data tables.

Underlying MIMIC-III and eICU records must be obtained through their authorized
PhysioNet access routes, including credentialing, training and the applicable
data-use agreement. The author-supplied MIMIC-IV extract is not redistributed.
Its extraction query and precise database/time-anchor provenance were not
verified; no generic public download is represented as an exact replacement.
There is no verified institutional Ruijin clinical reanalysis in this snapshot.

PhysioNet's MIMIC guidance treats derived datasets and models as sensitive and
directs their sharing through PhysioNet under the source agreement. Removing
direct identifiers does not make patient-level matrices, labels, predictions or
fitted model files suitable for a public GitHub repository. These artifacts are
excluded. The `.gitignore` also excludes generated result directories, because
they can contain both aggregate summaries and fitted parameters.

Official source and policy links (checked 9 September 2026):

- [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
- [MIMIC-IV data/model-sharing guidance](https://physionet.org/content/mimiciv/3.1/)
- [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/2.0/)
- [PhysioNet access FAQs](https://physionet.org/about/faqs/)
- [npj Digital Medicine reporting and code policy](https://www.nature.com/npjdigitalmed/for-authors-and-referees/about/editorial-policies/reporting-standards)

The MIMIC-IV policy link identifies the sharing guidance, not the verified
version of the author-supplied workbook. Code availability does not remove the
scientific limitations or imply the original warning-time claim was validated.
