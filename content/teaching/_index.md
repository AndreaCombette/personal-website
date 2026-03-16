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
      filters:
        folders:
          - teaching
    design:
      view: article-grid
      columns: 2
  - block: markdown
    content:
      title: "Lecture files"
      subtitle: ""
      text: |-
        Download course material below:

        - {{< icon name="code" pack="fas" >}} [Shallow Water Tutorial (Jupyter Notebook)](/uploads/teaching/CFD/tutorials/ShallowWaterTutorial.ipynb)
    design:
      columns: "1"
---