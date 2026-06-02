#include <Python.h>

#include <string>

#include "phonemizer.h"

static PyObject* ephonemizer_phonemize(PyObject*, PyObject* args, PyObject* kwargs) {
    const char* text = nullptr;
    const char* rules_path = nullptr;
    const char* list_path = nullptr;
    const char* dialect = "en-us";
    static const char* keywords[] = {"text", "rules_path", "list_path", "dialect", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "sss|s",
            const_cast<char**>(keywords),
            &text,
            &rules_path,
            &list_path,
            &dialect)) {
        return nullptr;
    }

    try {
        IPAPhonemizer phonemizer(rules_path, list_path, dialect);
        if (!phonemizer.isLoaded()) {
            PyErr_SetString(PyExc_RuntimeError, phonemizer.getError().c_str());
            return nullptr;
        }

        std::string result = phonemizer.phonemizeText(text);
        return PyUnicode_FromStringAndSize(result.data(), result.size());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown ephonemizer error");
        return nullptr;
    }
}

static PyMethodDef EPhonemizerMethods[] = {
    {"phonemize", reinterpret_cast<PyCFunction>(ephonemizer_phonemize), METH_VARARGS | METH_KEYWORDS, "Phonemize text with the KittenTTS C++ EPhonemizer."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef EPhonemizerModule = {
    PyModuleDef_HEAD_INIT,
    "_ephonemizer",
    "KittenTTS C++ EPhonemizer bindings.",
    -1,
    EPhonemizerMethods
};

PyMODINIT_FUNC PyInit__ephonemizer(void) {
    return PyModule_Create(&EPhonemizerModule);
}
