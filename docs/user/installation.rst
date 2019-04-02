Installation
=====================
Latest version (v0.1)
---------------------

Installing AdverTorch itself

We developed AdverTorch under Python 3.6 and PyTorch 1.0.0 & 0.4.1. To install AdverTorch, simply run

.. code-block:: bash

   pip install advertorch

or clone the repo and run

.. code-block:: bash

   python setup.py install

To install the package in "editable" mode:

.. code-block:: bash

   pip install -e .


Setting up the testing environments
-----------------------------------

Some attacks are tested against implementations in [Foolbox](https://github.com/bethgelab/foolbox) or [CleverHans](https://github.com/tensorflow/cleverhans) to ensure correctness. Currently, they are tested under the following versions of related libraries.

.. code-block:: bash

   conda install -c anaconda tensorflow-gpu==1.11.0
   pip install git+https://github.com/tensorflow/cleverhans.git@336b9f4ed95dccc7f0d12d338c2038c53786ab70
   pip install Keras==2.2.2
   pip install foolbox==1.3.2
