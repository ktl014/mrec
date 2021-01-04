====
mrec
====


.. image:: https://img.shields.io/pypi/v/mrec.svg
        :target: https://pypi.python.org/pypi/mrec

.. image:: https://img.shields.io/travis/ktl014/mrec.svg
        :target: https://travis-ci.com/ktl014/mrec

.. image:: https://readthedocs.org/projects/mrec/badge/?version=latest
        :target: https://mrec.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/ktl014/mrec/shield.svg
     :target: https://pyup.io/repos/github/ktl014/mrec/
     :alt: Updates



Medical Relation Extraction Classification (MREC) - MREC is a simple serverless Python application illustrating the
usage of an Amazon EC2 Instance to host a natural language processing application for classifying the relationships
of extracted medical terms a dataset of PubMed articles.


* Free software: MIT license
* Documentation: https://mrec.readthedocs.io.

Description
---------------

The MREC project showcases the power of natural language processing when it is enabled through cloud services, such
as Amazon.

Our selected model for classifying the medical relations was a Support Vector Machine. When given a sentence that
includes :code:`medical-term1` and :code:`medical-term2`, the classifier will return a predicted relationship such
as :code:`treats`.

The model was trained on over 1,500 medical sentences extracted from PubMed abstracts and relationships between
discrete medical terms.

The model is now deployed on an Amazon E2 instance with independent test data being piped into the application from
an Amazon S3 bucket.

Services and Features Used
***************************

Below are special services and features that we've incorporated into our workflow and application:

- Streamlit_ is a minimal framework for turning data scripts into sharable web apps.
- DVC_ is is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code.
- MLFlow_ is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry
- Amazon_EC2_instance_ is a web service that provides secure, resizable compute capacity in the cloud. It is designed to make web-scale cloud computing easier for developers.
- Amazon_S3_instance_ is an object storage service that offers industry-leading scalability, data availability, security, and performance.
- Amazon_RDS_instance_ makes it easy to set up, operate, and scale a relational database in the cloud.

.. _Streamlit: https://www.streamlit.io/
.. _DVC: https://dvc.org/
.. _MLFlow: https://www.mlflow.org/
.. _Amazon_EC2_instance: https://aws.amazon.com/ec2/?nc2=h_ql_prod_fs_ec2&ec2-whats-new.sort-by=item.additionalFields.postDateTime&ec2-whats-new.sort-order=desc
.. _Amazon_S3_instance: https://aws.amazon.com/s3/?nc2=h_ql_prod_fs_s3
.. _Amazon_RDS_instance: https://aws.amazon.com/rds/?nc2=h_ql_prod_fs_rds

Getting Started
---------------

These instructions will help you get a copy of the project up and running on your local machine for development and
testing purposes. See deployment for notes on how to deploy the project on a live system.

::

    $ git clone https://github.com/ktl014/mrec.git
    $ cd mrec
    $ git fetch --all --tags

Building
**********************

Install your local copy into a virtualenv. Assuming you have conda installed, this is how you set up
your fork for local development:

::

    $ conda env create -n mrec_env
    $ conda activate mrec_env
    $ pip3 install -r requirements.txt --user

Alternatively if you are using Docker, you can build teh image with the following,

::

    $ docker build -t mrec .

and then run it locally using,

::

    $ docker run -it -p 8501:8501 mrec

The final step is to pull the data and models from our S3 bucket.
Please make sure that you have access to the S3 credentials to pull the data.

::

    $ dvc pull

Usage
---------------
Instructions for using our project.

+---------------------------+---------------------------------------------------------------------------------+
| Step                      | Command                                                                         |
+===========================+=================================================================================+
| Train Model               | ::                                                                              |
|                           |                                                                                 |
|                           |  $ git checkout v1.2                                                            |
|                           |  $ dvc checkout                                                                 |
|                           |  $ python mrec/train_mrec.py                                                    |
+---------------------------+---------------------------------------------------------------------------------+
| Track Model Improvements  | ::                                                                              |
|                           |                                                                                 |
|                           |  # Change parameter in `params.yaml`                                            |
|                           |  $ dvc repro                                                                    |
|                           |  $ mlflow ui                                                                    |
|                           |  $ git add dvc.lock                                                             |
|                           |  $ git commit -m "Second model, tuned parameter"                                |
|                           |  $ git tag -a "v2.0" -m "model v2.0, tuned degree parameter"                    |
+---------------------------+---------------------------------------------------------------------------------+
| Access Shared Models/Data | ::                                                                              |
|                           |                                                                                 |
|                           |  # Download model                                                               |
|                           |  $ dvc get https://github.com/ktl014/mrec models/clean_data_model               |
|                           |  # alternatively for downloading data and code to a linked git&dvc project      |
|                           |  $ git pull && dvc pull                                                         |
+---------------------------+---------------------------------------------------------------------------------+
| Reproduce model           | ::                                                                              |
|                           |                                                                                 |
|                           |  $ dvc repro                                                                    |
+---------------------------+---------------------------------------------------------------------------------+

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
