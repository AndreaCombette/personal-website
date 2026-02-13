---
title: "A new initialisation to Control Gradients in Sinusoidal Neural network"
authors:
  - admin 
  - Antoine Venaille
  - Nelly Pustelnik

date: "2026-01-19T00:00:00"

doi: "https://doi.org/10.48550/arXiv.2512.06427"

# Schedule page publish date (NOT publication's date).
publishDate: "2026-01-19T00:00:00"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["paper-conference"]

# Publication name and optional abbreviated publication name.
publication: "A new initialisation to Control Gradients in Sinusoidal Neural network"
publication_short: ""

abstract: Proper initialisation strategy is of primary importance to mitigate gradient explosion or vanishing when training neural networks. Yet, the impact of initialisation parameters still lacks a precise theoretical understanding for several well-established architectures. Here, we propose a new initialisation for networks with sinusoidal activation functions such as SIREN, focusing on gradients control, their scaling with network depth, their impact on training and on generalization. To achieve this, we identify a closed-form expression for the initialisation of the parameters, differing from the original SIREN scheme. This expression is derived from fixed points obtained through the convergence of pre-activation distribution and the variance of Jacobian sequences. Controlling both gradients and targeting vanishing pre-activation helps preventing the emergence of inappropriate frequencies during estimation, thereby improving generalization. We further show that this initialisation strongly influences training dynamics through the Neural Tangent Kernel framework (NTK). Finally, we benchmark SIREN with the proposed initialisation against the original scheme and other baselines on function fitting and image reconstruction. The new initialisation consistently outperforms state-of-the-art methods across a wide range of reconstruction tasks, including those involving physics-informed neural networks.
# Summary. An optional shortened abstract.
summary:
tags:
  - Machine Learning
  - Learning Theory
featured: false

url_pdf: "https://arxiv.org/abs/2512.06427"
# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides:
---
<iframe
  src="SIREN-init.pdf"
  style="
    width: 100%;
    height: 800px;
    border: 5px solid #000;      /* black border, 5px thick */
    border-radius: 12px;         /* rounded corners */
    overflow: hidden;            /* clip any overflowing PDF content */
  "
  webkitallowfullscreen
  mozallowfullscreen
  allowfullscreen>
</iframe>