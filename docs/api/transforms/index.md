# Transforms { #mlx_graphs.transforms }

<!-- api mlx_graphs.transforms -->
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
<!-- /api -->

## Point clouds and meshes

Transforms of node positions `pos` and mesh triangles `face`. Random
transforms draw from `mx.random`.

<!-- api mlx_graphs.transforms -->
SamplePoints
FixedPoints
NormalizeScale
Center
LinearTransformation
RandomJitter
RandomRotate
RandomScale
FaceToEdge
Cartesian
Distance
Polar
PointPairFeatures
<!-- /api -->
