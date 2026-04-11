# NeurIPS 2026 Source Audit

Audit date: 2026-04-03

This audit used official NeurIPS sources first, and only official NeurIPS URLs are cited below.

## Official sources used

1. Call for Papers 2026  
   URL: https://neurips.cc/Conferences/2026/CallForPapers

   Established:
   - NeurIPS 2026 has an official CFP page.
   - The page links to the official 2026 formatting package hosted on `media.neurips.cc`.

2. Official formatting package  
   URL: https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

   Established:
   - The official current-year template package was reachable and downloaded successfully.
   - The package contains:
     - `neurips_2026.tex`
     - `neurips_2026.sty`
     - `checklist.tex`
   - The repo now uses the extracted official `neurips_2026.sty`.

3. Main Track Handbook 2026  
   URL: https://neurips.cc/Conferences/2026/MainTrackHandbook

   Established:
   - Authors must use the current year's LaTeX style file.
   - Style violations or page-limit violations may be desk rejected.
   - The handbook also contains code, checklist, compute-reporting, and LLM-policy guidance.

4. NeurIPS Paper Checklist Guidelines  
   URL: https://neurips.cc/public/guides/PaperChecklist

   Established:
   - The checklist is required.
   - Omitting the checklist can cause desk rejection.
   - The required PDF order is:
     1. paper
     2. optional technical appendix
     3. checklist
   - The checklist does not count toward the page limit.

5. Dates and Deadlines 2026  
   URL: https://neurips.cc/Conferences/2026/Dates

   Established:
   - The official 2026 conference dates are public.
   - Workshops are scheduled for December 11-12, 2026.
   - The page lists main-track dates, but does not provide workshop-specific CFP text for this paper.

6. Downloads 2026  
   URL: https://neurips.cc/Downloads/2026

   Established:
   - There is an official 2026 downloads page.
   - As inspected during this audit, it did not expose a concrete 2026 workshop list or workshop CFP page for targeting this paper.

## Template and formatting findings

- Official current-year style package fetched successfully: yes
- Local fallback required: no
- Current-year style now used in the repo: yes (`neurips_2026.sty`)
- Current-year checklist source fetched successfully: yes

## Workshop-process findings

- Official NeurIPS 2026 workshop dates are public: yes
- Official NeurIPS 2026 workshop list/public workshop homepages were located on neurips.cc during this audit: no
- Official workshop-specific CFP for a target workshop was located: no

This means the manuscript was adapted under the user-specified Case B rule:
- target a workshop theme family rather than inventing a specific workshop title or CFP.

## Remaining ambiguity

- Because official 2026 workshop pages were not public in the inspected official sources, workshop-specific formatting or single-blind/double-blind submission rules could not be verified.
- The final manuscript is therefore prepared in official NeurIPS 2026 style as a workshop-ready default artifact, with theme-family targeting documented separately.
