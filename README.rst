CDE-conformal and ``local_conformal`` package
=============================================


.. |Git Actions| image:: https://github.com/benjaminleroy/CDE-conformal/workflows/pytest%20check%20and%20code%20coverage/badge.svg
  :target: https://github.com/benjaminleroy/CDE-conformal/actions

.. |Codecov| image:: https://codecov.io/gh/benjaminleroy/CDE-conformal/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/benjaminleroy/CDE-conformal




Using ``pytest``
===================

To test the code, you'll need to first download ``pytest``

.. code-block:: bash

  conda install pytest # running tests
  conda install pytest-cov # code coverage


Then you can run ``pytest`` in the project directory to test of the functions,
and ``pytest --cov=local_conformal tests/`` to get the coverage (run in
``local_conformal`` directory).


