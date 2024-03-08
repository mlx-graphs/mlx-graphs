{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
	:members:
	:show-inheritance:

	{% block methods %}

	{% if methods %}
	.. rubric:: {{ _('Methods') }}

	.. autosummary::
	{% for item in methods %}
	  {%- if item not in inherited_members and item != '__init__' %}
	     ~{{ name }}.{{ item }}
	  {%- endif %}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{% block attributes %}
	{% if attributes %}
	.. rubric:: {{ ('Attributes') }}

	.. autosummary::
	{% for item in attributes %}
	  ~{{ name }}.{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}
