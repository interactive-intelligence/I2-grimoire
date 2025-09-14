---
title: I2 Grimoire
layout: home
nav_order: 1
---
# I2 Grimoire 📚

## [Main PDF](./I2%20Grimoire.pdf)

## Units

<ul>
{% for file in site.static_files %}
  {% if file.path contains '/units/' and file.extname == ".pdf" %}
    <li><a href="{{ file.path | relative_url }}">{{ file.name | replace: '.pdf', '' }}</a></li>
  {% endif %}
{% endfor %}
</ul>

## NEW: Experimental web version
Use the navigation bar to explore the experimental web version
