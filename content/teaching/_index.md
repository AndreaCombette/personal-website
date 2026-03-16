---
title: Teaching
summary: My courses
type: landing

cascade:
  - _target:
      kind: page
    params:
      show_breadcrumb: true

sections:
  - block: collection
    id: teaching
    content:
      title: Teaching
      text: ""
      filters:
        folders:
          - teaching
    design:
      view: article-grid
      columns: 2

  - block: markdown
    id: resources
    content:
      title: Downloadable material
      subtitle: Exercise sheets, notes, and solutions
      text: |-
        ### Computational Method For Geophsyics and Astronomy
        - <i class="fa-solid fa-code"></i>[Exercise Sheet 1](/static/uploads/teaching/CFD/tutorials/ShallowWaterTutorial.ipynb)
    design:
      columns: "1"
---