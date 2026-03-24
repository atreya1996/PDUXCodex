# UI Pull Request Checklist

Use this checklist for **every PR that changes the dashboard UI**. Complete each item before requesting review.

## Required checks

- [ ] Uses design tokens from `skills/streamlit_ui_system.md`.
- [ ] Sidebar + tab layout is preserved.
- [ ] Interview card click opens a modal overlay (session-state driven).
- [ ] No raw JSON is exposed in main UI views.
- [ ] Contrast/readability is validated for cards, labels, and buttons.
- [ ] Charts have clear labels and readable fonts.
- [ ] Spacing follows the 8px grid with a minimum of 16px padding.
- [ ] Screenshots are included for:
  - [ ] Overview tab
  - [ ] Cohorts tab
  - [ ] Personas tab
  - [ ] Interviews tab
  - [ ] Interview modal open state

## Review note

Add a short note in the PR description describing any checklist item marked N/A and why.
