Project Short Description

Project consist of two parts: Classification and Optimization.
First system executed to find classification boundaries and then, given a new instance and desired class system finds
closest area (bounded by hyperplanes) and point in this area which is closest to the given instance meaning that
the system provides advice how to change input instance to reach desired class with minimal changes.
In GT domain this means: how to change NFG matrix in order to influence payer(s) to choose
desired (by game experimentator) action.

Classification
Given a dataset of any domain the system knows to build classification boundaries with hyperplanes.
Introduced layered layered boundary definition:
    system searches in space for hyperplanes position to define boundaries with predefined accuracy 
    (when separating instances to areas we may allow some percentage of instances to be of wrong class).
    if there is(are) area(s) (bounded by hyperplanes) where it was not possible to separate with predefined accuracy
    then such (these) areas extracted and search executed on it separately. This can be executed on any depth level.
    Finally the hyperplane boundary state is saved for future use during optimization phase
The final product of this phase is list of areas - definitions of polytopes bounded by hyperplanes.
As search optimization technique system uses Reinforcement Learning approach, although this is not the point of this
project but just a tool: we could choose any kind of approach (like Genetic Algorithm, Random Walk and, etc).
As I am interesting in AI and machine learning I decided to try this one. I have used gym and keras libraries for this project

Optimization
Given a new instance and desired class definition the system finds out all possible areas for that class and finds out
the point from these areas which is the closest to the given instance.
There are several modes here:
 - border point - point on the border of polytope area
 - features-bound point - point within the max and min values of already seen features
 - in between - point in the middle of border point and existing point from polytope closest to the given new instance 
As a performance test for this part: before optimization system trains several classic machine learning classifiers 
(like nearest neighbors, svm, random forest, simple neural network) with provided dataset and after closest points found
system checks how good these classifiers identify these new points.
For finding closest point from point to a polytope area, used GEKKA optimization library which can solve large-scale
algebraic equations. 