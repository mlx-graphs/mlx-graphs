{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
	:members:
	:show-inheritance:

	{% block methods %}
	{% if '__call__' in members %}
	.. automethod:: __call__
	{% endif %}

	{% set own_methods = [] %}
	{% for item in methods %}
	  {%- if item not in inherited_members and item != '__init__' %}
	    {%- set _ = own_methods.append(item) %}
	  {%- endif %}
	{%- endfor %}
	{% if own_methods %}
	.. rubric:: {{ _('Methods') }}

	.. autosummary::
	{% for item in own_methods %}
	     ~{{ name }}.{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}
