# Streamlit UI System Skill

## 1. Purpose

Production-grade Streamlit analytics UX.

## 2. Design system tokens

- Background: `#0B0F19`
- Card: `#111827`
- Primary: `#4F46E5`
- Text primary: `#F9FAFB`
- Text secondary: `#9CA3AF`
- Spacing: 8px grid baseline
- Component padding: minimum 16px

## 3. Required layout structure

- Sidebar: upload + filters
- Main tabs:
  - Overview
  - Cohorts
  - Personas
  - Interviews

## 4. Tab-by-tab UX contract

### Overview

- KPI cards at the top
- Charts in the middle
- Summaries at the bottom

### Cohorts

- Filter snapshot at top
- Clean grouped tables for cohort breakdowns

### Personas

- Card-based presentation with:
  - Persona name
  - Persona size
  - Persona description

### Interviews

- Card grid for interview items
- Modal overlay behavior for interview detail

## 5. Interaction rules

- Clicking an interview card opens a modal via Streamlit session state
- No page navigation for interview detail
- No raw JSON in the primary UI

## 6. Streamlit implementation constraints

- Must use `st.columns`
- Must use `st.container`
- Must use `st.expander`
- Modal must be simulated with a session-state overlay pattern

## 7. Chart standards

- Clear labels
- Minimal gridlines
- Readable text sizing and contrast

## 8. Accessibility/readability checks

- Contrast checks
- Text hierarchy checks
- Spacing consistency checks

## 9. Definition of done checklist

- Visual hierarchy is clear and consistent
- Layout remains responsive across viewport sizes
- Modal behavior works reliably through session state
- UI remains uncluttered and scannable
