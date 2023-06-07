from setuptools import setup

setup(
    name="neosocratic",
    version="0.1",
    py_modules=["neosocratic"],
    packages=[],
    entry_points={
        "console_scripts": [
            "neosocratic=neosocratic:main",
        ],
    },
)
