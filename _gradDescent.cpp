#include <Python.h>
#include <numpy/arrayobject.h>
#include "gradDescent.h"


static char module_docstring[] =
    "This module provides an interface for solving linear systems using steepest
     gradient descent.";

static char gradDescent_docstring[]=
    "Calculates the solution of a linear system using the steepest descent
    algorithm.";

static PyObject* gradDescent_gradDescent(PyObject *self, PyObject *args);

static PyMethodDef moduel_methods[] = {
    {"gradDescent", gradDescent_gradDescent, METH_VARARGS, chi2_docstring}.
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_gradDescent(void)
{
  PyObject *m = Py_InitModule3("_gradDescent", module_methods, module_docstring);
  if (m == NULL)
    return;

  import_array();
}

static PyObject* gradDescent_gradDescent(PyObject *self, PyObject *args)
{

  PyObject *a_obj, *b_obj;
  /* Parse Input Data */
  if(!PyArg_ParseTuple(args,"OO", &a_obj, &b_obj))
    return NULL;

  /* Place input into correct numpy data types (a = db matrix, b = db vector) */
  PyObject *a_matrix = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *b_vec = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);

  /* If numpy array doesn't work throw exception*/
  if(a_matrix == NULL || b_vec == NULL)
  {
    Py_XDECREF(a_matrix);
    Py_XDECREF(b_vec);
    return NULL;
  }

  /* Check that we have a matrix and an array */
  if(PyArray_NDIM(a_matrix) != 2 || PyArray_NDIM(b_vec) != 1)
  {
    Py_XDECREF(a_matrix);
    Py_XDECREF(b_vec);
    return NULL;
  }

  /* Shape of matrix a */
  int rows = (int)PyArray_DIM(a_matrix,0);
  int cols = (int)PyArray_DIM(a_matrix,1);
  int len = (int)PyArray_DIM(b_vec,0);

  /*Grab pointers to ndarrays */
  double *a = (double*)PyArray_DATA(a_matrix);
  double *b = (double*)PyArray_DATA(b_vec);

  /*Execute external gradDescent*/
  double *x = gradDescent(a,b,rows,cols, len);

  /* Deallocate References */
  Py_DECREF(a_matrix);
  Py_DECREF(b_vec);

  if (x == NULL)
  {
    PyErr_SetString(PyExc_RuntimeError, "Gradient Descent Failed." );
    return NULL;
  }
  
  PyObject *sol = Py_BuildValue("N", x);
  return sol;
}
