{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
	:members:
	:special-members: __cat_dim__, __inc__

	{% block methods %}

	{% if methods %}
	.. rubric:: {{ _('Methods') }}

	.. autosummary::
	{% for item in all_methods %}
		{%- if not item.startswith('_') or item in ['__cat_dim__','__inc__',] %}
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
