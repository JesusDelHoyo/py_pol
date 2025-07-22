=======
History
=======

1.3.0
-----------------------
* Adition of ret as variable name for retardance in all retarder creation methods.
* Adition of general, diattenuator_general and retarder_general methods which accept both characteristic angles and azimuth_ellipticity methods.
* Addition of biaxial_material for creating retarders from their ordinary and extraordinary refractive indices.
* Addition of Liquid crystal advanced example using different biaxial layer models.
* Citation file.
* Bug fix of multiplication of elements by arrays.
* Fix in the definition of righ-handed and left-handed in the polarization ellipse representation.
* Other bugfixes.

1.2.0
-----------------------
* Addition of general creation methods that encompases both azimuth_ellipticity and charac_angles methods (which are maintained for retrocompatibility).
* Added ret as parameters of retarders for homogeneity in naming (R is maintained for retrocompatibility).
* New circular diattenuators and retarders methods.
* Limits for alpha, delay, azimuth, ellipticity and retardance values are optional through the force_limits variable in creation methods, which is False by default.
* New liquid crystal advanced exmple.
* Bugfixes.

1.1.4
-----------------------
* bugfix Jones_matrix.parameters.transmission
* Remove the global phase of a diattenuattor before checking if it is a diattenuator
* bugfix Jones_matrix.analysis.decompose_pure
* Include amplitude_phase drawing
* Change the representation origin to 'lower'

1.1.3
-----------------------
* bugfix zeroslike
* bugfix not taking shape = False as flatten the object.

1.1.2
-----------------------
* bugfix Jones_vector.parameters.global_phase

1.1.1
-----------------------
* from_list can also be used with lists of Py_pol objects, lists and tuples. Also, all elemeents may have more than one elemnt.
* minor bugfixes


1.1.0 (2022-3-31)
-------------------
* Added base class Py_pol
* Objects are now iterable
* 3D drawings changed to Plotly
* New density and existence methods


1.0.4 (2022-02-07)
------------------
* Bug fix related to variable limits

1.0.4 (2021-07-19)
------------------
* Bug fixes


1.0.3 (2021-01-22)
------------------
* Bug fixes


1.0.2 (2020-07-04)
--------------------
* Implemented workaround of axis_equal issue.


1.0.0 (2020-06-04)
-------------------
py_pol multidimensional. Alpha state

This is a big overhaul with many changes. All of them are based on the possibility of storing several vector/matrices in the same object. This reduces significantly the time required to perform the same operation to multiple vectors/matrices, using numpy methods instead of for loops. We have calculated that the reduction is around one order of magnitude.

New methods have been introduced. First, methods available for Mueller / Stoes modules have been created also for Jones (when possible). Also, some bugs and errors in the calculations have been solved.

Finally, some method and argument names have been changed to be consistent between different classes. Also, the default value of arguments with the same name have also been unified.

The biggest TO DO we have are tests. Right now, we only have tests for the Jones_vector class. However, we thought that it would be useful to release this version so the community can use it.

NOTE: Due to the change of argument and method names, this version is not compatible with the previous ones.


0.2.2 (2019-09-04)
------------------
* Bug fixes.


0.2.1 (2019-09-04)
------------------
* Bug fixes.
* Solve incidents.
* Start to homogenize structures for both Jones and Stokes.


0.2.0 (2019-05-25)
------------------
pre-alpha state

* Upgrade to Python 3
* Stable version including tests


0.1.5 (2019-02-25)
------------------
* Jones_vector, Jones_matrix, Stokes works.
* Jones_vector: simplify function to represent better Jones vectors.
* tests drawing: Made tests for drawing

* Mueller is in progress.
* Functions = 9/10
* Documentation = 8/10
* Tutorial = 8/10.
* Examples = 8/10.
* Tests = 8/10
* Drawing = 10/10. Finished. Polarization ellipse for Jones and Stokes (partially random). Stokes on Poincaré sphere.


0.1.4 (2019-02-03)
------------------
* bug fixes


0.1.3 (2019-01-22)
------------------
* Fixed axis_equal issue.
* Jones_vector, Jones_matrix, Stokes works.
* Mueller is in progress.
* Functions = 9/10
* Documentation = 8/10
* Tutorial = 7/10.
* Examples = 6/10.
* Drawing = 0/10.


0.1.1 (2018-12-22)
------------------
* First release on PyPI in alpha state.


0.0.0 (2018-11-22)
------------------
First implementation of py_pol.
