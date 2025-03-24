---
title: I2 Grimoire
layout: minimal
---
# I2 Grimoire 📚
<a style="font-size: 20px" href="./I2%20Grimoire.pdf">Link to Main PDF</a>

## Units

<ul>
{% for file in site.static_files %}
  {% if file.path contains '/units/' and file.extname == ".pdf" %}
    <li><a href="{{ file.path | relative_url }}">{{ file.name | replace: '.pdf', '' }}</a></li>
  {% endif %}
{% endfor %}
</ul>
