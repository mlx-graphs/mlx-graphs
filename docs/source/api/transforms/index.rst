.. _transforms:

Transforms
===============

.. currentmodule:: mlx_graphs.transforms


.. autosummary::
	:toctree: _autosummary
	:template: transform-class-template.rst

	BaseTransform
	Compose
	FeaturesNormalizedTransform
	ToUndirected
	AddSelfLoops
	RemoveSelfLoops
	RemoveIsolatedNodes
	LargestConnectedComponents
	RandomNodeSplit
	RandomLinkSplit
	OneHotDegree
	Constant
	AddRandomWalkPE
	GCNNorm
	AddRemainingSelfLoops
	RemoveDuplicatedEdges
	IndexToMask
	MaskToIndex
	KNNGraph
	RadiusGraph
	NormalizeFeatures
	TargetIndegree
	LocalDegreeProfile
	AddLaplacianEigenvectorPE
	SVDFeatureReduction
	SIGN
	FeaturePropagation
	VirtualNode
	TwoHop
	LineGraph
	ToDense
	RemoveTrainingClasses
