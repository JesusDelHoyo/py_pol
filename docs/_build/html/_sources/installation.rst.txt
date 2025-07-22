.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Python polarization package py_pol, run this command in your terminal:

.. code-block:: console

    $ pip install py-pol

This is the preferred method to install Python polarization, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/master/starting/installation/

**Note:** If py_pol is installed using Python 3.9+, pip may encounter an error trying to install h5py. A workaround of this problem is downloading the module hickle from its `GitHub webpage`_. Then, the user may have to edit the requirements.txt file from hicle to remove a limitation on the upper version of h5py. Then, the user must install this modified version of hickle. Then, the user can install py_pol without any other known problem.

.. _GitHub webpage: https://github.com/telegraphic/hickle

From sources
------------

The sources for Python polarization can be downloaded from the `Bitbucket repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@bitbucket.org:optbrea/py_pol.git
    $ git clone https://optbrea@bitbucket.org/optbrea/py_pol.git



Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Bitbucket repo: https://bitbucket.org/optbrea/py_pol/src/master/
