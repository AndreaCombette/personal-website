---
# Leave the homepage title empty to use the site title
title: ""
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: "6rem"

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
      text: ""
    design:
      css_class: dark
      background:
        color: white
        image:
          # Add your image background to `assets/media/`.
          filename: back1.png
          filters:
            brightness: 1.0
          size: cover
          position: center
          parallax: false
          caption: "Image credit: [**Peony, 2017**](James Lahey, Mixed Media on Canvas)"
          focal_point: Right

  - block: markdown
    content:
      title: "About"
      subtitle: ""
      text: |-
        My work has involved various projects centered on temporal and spatial simulations, with a particular focus on stochastic and physical phenomena. On this website, I will present several significant projects broadly covering the field of computational physics. We will explore the numerical resolution of various types of partial differential equations (PDEs) applied to a wide range of physical phenomena, as well as the use of Monte Carlo simulations for modeling both quantum and classical systems.

        In addition, I have a strong interest in machine learning. Consequently, I will showcase various projects in this field, ranging from biological applications to more physics-oriented topics, such as SIREN and LSTM networks. These projects will cover classical regression and classification problems, as well as more complex topics, including the study of generative adversarial networks (GANs) and diffusion models.
        {{% callout note %}}
        This site is still under construction and is maintained and complete during my free time, feel free to contact me if you have any suggestion or question.
        {{% /callout %}}
    design:
      columns: "1"
  - block: collection
    id: publication
    content:
      title: Recent Reports & Publications
      text: ""
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      view: citation
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ""
      text: ""
      # Page type to display. E.g. post, talk, publication...
      page_type: post
      # Choose how many pages you would like to display (0 = all pages)
      count: 3
      # Filter on criteria
      filters:
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      # Reduce spacing
      spacing:
        padding: [0, 0, 0, 0]
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      view: article-grid
      columns: 2
---
