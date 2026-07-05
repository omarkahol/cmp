---
noteId: "fbc3f2f0789111f18c2a51b1936a4396"
tags: []

---

# CMP++ Doxygen Style & Best Practices Guide

This document establishes the styling and documentation standards for the **CMP++** codebase to ensure the generated Doxygen website remains clean, accessible, and mathematically rigorous.

---

## 1. Documentation Structure & Categorization
All new classes and functions must be grouped into the following architectural sections:
1. **Surrogate Modeling & Dimensionality Reduction**
2. **Sensitivity Analysis**
3. **Sampling & Inference Methods**
4. **Probability Distributions, Priors, & Kernels**
5. **Classifiers**
6. **Clustering Methods**
7. **Core Utilities & Integration**

When adding a new class, ensure it is added to the corresponding category on the main landing page config ([Mainpage.h](file:///Users/omarkahol/opt/CMP++/include/Mainpage.h)) with a clear, one-sentence mathematical summary.

---

## 2. Mathematical Typesetting
CMP++ uses **MathJax** via the `USE_MATHJAX = YES` directive in the `Doxyfile`.
- **Inline Formulas**: Use `\f$` as the delimiter.
  - Example: `\f$y = f(x) + \varepsilon\f$` renders as \(y = f(x) + \varepsilon\).
- **Block Equations**: Use `\f[` and `\f]` as delimiters.
  - Example:
    ```cpp
    * \f[
    * \theta_{\text{real}, k} = \exp\left(\theta_{\text{opt}, k}\right)
    * \f]
    ```
- **Guidelines**:
  - Always use standard LaTeX notation.
  - Do not use markdown style single/double dollar signs (`$` or `$$`) as Doxygen does not reliably parse them in all settings and they can trigger unknown command warnings (e.g. `\mid` warning).
  - Use `|` or `\vert` instead of `\mid` to prevent Doxygen parser warnings outside of formal LaTeX math environments.

---

## 3. Image & Diagram Embedding
To include visual diagrams or pipeline illustrations:
1. **File Format**: Prefer high-resolution PNG or SVG for web display.
2. **Configuration**: Every image must be registered in the `Doxyfile` under the `HTML_EXTRA_FILES` configuration. This forces Doxygen to copy the image files directly into the build HTML folder.
   - Example:
     ```
     HTML_EXTRA_FILES = main.pdf docs/images/my_diagram.png
     ```
3. **Doxygen Integration**: Use the standard Doxygen `\image` command.
   - **For Logos/Decorative Elements** (No Caption Text):
     ```cpp
     * \image html logo.png "" width=550px
     ```
   - **For Captioned Figures**:
     ```cpp
     * \image html gp_posterior.png "Gaussian Process Predictive Distribution"
     ```

---

## 4. Light & Dark Theme Compatibility
CMP++ uses the custom stylesheet [custom_style.css](file:///Users/omarkahol/opt/CMP++/custom_style.css) alongside the `doxygen-awesome.css` theme.
- **Color Variables**: Never hard-code background colors (e.g., `#ffffff`, `#000000`) for structural elements like math blocks (`.formulaDsp`) or code blocks. This breaks dark-mode styling.
- **CSS Variables**: Use Doxygen Awesome's native variables to automatically adjust colors:
  - Backgrounds: `var(--bg-color-second)`
  - Text colors: `var(--page-foreground-color)`
  - Borders: `var(--separator-color)`
