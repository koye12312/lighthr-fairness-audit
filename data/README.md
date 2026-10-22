# Data

The LightHR dataset is synthetic and is not included in this repository.

To run the notebooks, put the file here as `data/LightHR_dataset.csv`. It has 8,800 rows and these 15 columns:

| Column | Example |
|---|---|
| Gender | Female, Male, Non-binary, Trans |
| Age at time of application | 40–49 |
| Year of Application | 1998–2022 |
| A-levels scores | 40–89 |
| Education | Arts, Business, Law, STEM |
| University Name | about 150 universities, some spelled several ways (see `lighthr/universities.py`) |
| Address | UK county |
| Current Employment | Salaried, Contractual, Self Business, or empty (unemployed) |
| Current Employment Sector | IT, Health, … or empty |
| Current Annual Income | 0–149,988 |
| References Submitted | 0 or 1 |
| Refree 1, Refree 2, Refree 3 | `EmploymentType:Sector`, or `NA` |
| Was Hired by a Company within 6 months | 0 or 1 (the label) |
